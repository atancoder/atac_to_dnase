from typing import List, Optional, Tuple, Dict, Set, cast

import numpy as np
import pandas as pd
import pyBigWig
import pysam
import torch

from .utils import (
    BED3_COLS,
    REGION_SLOP,
    estimate_bigwig_total_reads,
    one_hot_encode_dna,
    compute_RPM
)

def get_features(regions_df: pd.DataFrame, atac_bw_file: str, fasta_file: str) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Gets features but also returns a new regions_df, which filters out skipped regions
    """
    X = []
    regions_skipped = set()
    with pysam.FastaFile(fasta_file) as fasta:
        with pyBigWig.open(atac_bw_file) as atac_bw:
            atac_total_reads = estimate_bigwig_total_reads(atac_bw)
            for idx, row in regions_df.iterrows():
                chrom, start, end = row[BED3_COLS]
                # Shouldn't have to worry about going over chromosome boundaries
                start -= REGION_SLOP
                end += REGION_SLOP
                features = _gen_features(
                    chrom,
                    start,
                    end,
                    fasta,
                    atac_bw,
                    atac_total_reads,
                )
                if features is None:
                    regions_skipped.add(idx)
                    continue
                X.append(features)
    print(f"Skipping {len(regions_skipped)} regions due to lack of coverage or sequence")
    X = torch.tensor(np.array(X), dtype=torch.float32)
    filtered_regions = regions_df[~regions_df.index.isin(regions_skipped)]
    return X, filtered_regions

def get_labels(
    regions_df: pd.DataFrame, dnase_bw_file: str
) -> torch.Tensor:
    Y = []
    with pyBigWig.open(dnase_bw_file) as dnase_bw:
        dnase_total_reads = estimate_bigwig_total_reads(dnase_bw)
        for _, row in regions_df.iterrows():
            chrom, start, end = row[BED3_COLS]
            # Shouldn't have to worry about going over chromosome boundaries
            start -= REGION_SLOP
            end += REGION_SLOP
            label = _gen_labels(
                chrom,
                start,
                end,
                dnase_bw,
                dnase_total_reads,
            )
            Y.append(label)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    return Y

def normalize_features_and_labels(X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Returns the stats used to normalize
    """
    atac_signal = X[:,:,4].view(-1)
    dnase_signal = Y.view(-1)
    combined_signal = torch.cat((atac_signal, dnase_signal), dim=0)
    mean = combined_signal.mean().item()
    std = combined_signal.std().item()
    stats = {"std": std, "mean": mean}
    X = normalize_features(X, stats)
    Y = normalize_labels(Y, stats)
    return X, Y, stats

def normalize_features(X: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    mean, std = stats["mean"], stats["std"]
    X = X.clone()
    X[:,:,4] = (X[:,:,4] - mean) / std
    return X

def normalize_labels(Y: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    mean, std = stats["mean"], stats["std"]
    return (Y - mean) / std

def denormalize_labels(Y: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    mean, std = stats["mean"], stats["std"]
    Y = (Y * std) + mean
    return Y


def _gen_labels(
    chrom: str,
    start: int,
    end: int,
    dnase_bw: pyBigWig.pyBigWig,
    dnase_total_reads: int,
) -> np.ndarray:
    return cast(np.ndarray, compute_RPM(np.array(dnase_bw.values(chrom, start, end + 1)), dnase_total_reads))

def _gen_features(
    chrom: str,
    start: int,
    end: int,
    fasta: pysam.FastaFile,
    atac_bw: pyBigWig.pyBigWig,
    atac_total_reads: int,
) -> Optional[np.ndarray]:
    seq = fasta.fetch(chrom, start, end + 1)
    if not isinstance(seq, str):
        return None

    ohe = one_hot_encode_dna(seq)
    atac_signal = np.array(atac_bw.values(chrom, start, end + 1)) / atac_total_reads
    if sum(atac_signal) == 0:
        return None

    combined_features = np.hstack((ohe, atac_signal.reshape(-1, 1)))
    return combined_features