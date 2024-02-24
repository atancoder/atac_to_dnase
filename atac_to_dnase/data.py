from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyBigWig
import pysam
import torch
from torch.utils.data import DataLoader, TensorDataset

from .utils import (
    BED3_COLS,
    REGION_SLOP,
    estimate_bigwig_total_reads,
    one_hot_encode_dna,
)


def gen_features_and_label(
    chrom: str,
    start: int,
    end: int,
    fasta: pysam.FastaFile,
    atac_bw: pyBigWig.pyBigWig,
    atac_total_reads: int,
    dnase_bw: pyBigWig.pyBigWig,
    dnase_total_reads: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    seq = fasta.fetch(chrom, start, end + 1)
    if not isinstance(seq, str):
        return None

    ohe = one_hot_encode_dna(seq)
    atac_signal = np.array(atac_bw.values(chrom, start, end + 1)) / atac_total_reads
    dnase_signal = np.array(dnase_bw.values(chrom, start, end + 1)) / dnase_total_reads

    if sum(atac_signal) == 0 or sum(dnase_signal) == 0:
        # Only find data where we have signal in both
        return None

    combined_features = np.hstack((ohe, atac_signal.reshape(-1, 1)))
    return combined_features, dnase_signal


def get_dataset(
    regions_tsv: str, atac_bw_file: str, dnase_bw_file: str, fasta_file: str
) -> TensorDataset:
    regions_df = pd.read_csv(regions_tsv, sep="\t")
    X, Y = [], []
    regions_skipped = 0
    with pysam.FastaFile(fasta_file) as fasta:
        with pyBigWig.open(atac_bw_file) as atac_bw:
            with pyBigWig.open(dnase_bw_file) as dnase_bw:
                atac_total_reads = estimate_bigwig_total_reads(atac_bw)
                dnase_total_reads = estimate_bigwig_total_reads(dnase_bw)
                for _, row in regions_df.iterrows():
                    chrom, start, end = row[BED3_COLS]
                    # Shouldn't have to worry about going over chromosome boundaries
                    start -= REGION_SLOP
                    end += REGION_SLOP
                    result = gen_features_and_label(
                        chrom,
                        start,
                        end,
                        fasta,
                        atac_bw,
                        atac_total_reads,
                        dnase_bw,
                        dnase_total_reads,
                    )
                    if not result:
                        regions_skipped += 1
                        continue

                    features, label = result
                    X.append(features)
                    Y.append(label)
    print(f"Skipping {regions_skipped} regions due to lack of coverage or sequence")
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    return TensorDataset(X, Y)


def get_dataloader(
    dataset: TensorDataset, atac_bw_file: str, dnase_bw_file: str, batch_size: int
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size)
