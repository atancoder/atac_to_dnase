import math
import os
from typing import Dict, List, Optional, Set, Tuple, cast

import numpy as np
import pandas as pd
import pyBigWig
import pysam
import torch
import ast
from sympy import sequence

from .utils import (
    BED3_COLS,
    NORMAL_CHROMOSOMES,
    one_hot_encode_dna,
)

CACHED_X_FILE = "features.pt"
CACHED_Y_FILE = "labels.pt"


def split_into_fixed_region_sizes(
    abc_peaks_df: pd.DataFrame, region_size: int, region_slop: int
) -> pd.DataFrame:
    if (abc_peaks_df["end"] - abc_peaks_df["start"] < region_size).sum() > 0:
        raise Exception(f"There are abc regions that are smaller than {region_size}")

    large_regions = abc_peaks_df[
        abc_peaks_df["end"] - abc_peaks_df["start"] > region_size
    ]
    split_regions = _split_large_regions(large_regions, region_size)

    fixed_regions = abc_peaks_df[
        abc_peaks_df["end"] - abc_peaks_df["start"] == region_size
    ].copy()
    fixed_regions = pd.concat([fixed_regions, split_regions], ignore_index=True)
    chrom_order = abc_peaks_df["chrom"].unique().tolist()
    fixed_regions = _bedtools_sort(fixed_regions, chrom_order)

    # Shouldn't have to worry about going over boundaries??
    fixed_regions["start"] -= region_slop
    fixed_regions["end"] += region_slop
    fixed_regions["region_slop"] = region_slop
    return fixed_regions


def get_region_features(
    regions_df: pd.DataFrame, atac_bw_file: str, dnase_bw_file: str, fasta_file: str
) -> pd.DataFrame:
    regions_skipped = set()
    fasta = pysam.FastaFile(fasta_file)
    atac_bw = pyBigWig.open(atac_bw_file)
    dnase_bw = pyBigWig.open(dnase_bw_file)
    regions_df["ATAC"], regions_df["DNASE"] = None, None
    for idx, row in regions_df.iterrows():
        idx = cast(int, idx)
        chrom, start, end = row[BED3_COLS]
        atac_signal = get_coverage(chrom, start, end, atac_bw)
        dnase_signal = get_coverage(chrom, start, end, dnase_bw)
        sequence = _get_sequence(chrom, start, end, fasta)
        if sum(atac_signal) == 0 or sum(dnase_signal) == 0 or not sequence:
            regions_skipped.add(idx)
            continue
        regions_df.at[idx, "ATAC"] = atac_signal
        regions_df.at[idx, "DNASE"] = dnase_signal
        regions_df.at[idx, "SEQ"] = sequence

    fasta.close()
    atac_bw.close()
    dnase_bw.close()
    print(
        f"Skipping {len(regions_skipped)} regions due to lack of coverage or sequence"
    )
    filtered_regions = regions_df[~regions_df.index.isin(regions_skipped)]
    return filtered_regions


def load_features_and_labels(
    regions_file: str, cache_dir: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    result = _check_cache(regions_file, cache_dir)
    if result:
        return result[0], result[1]
    print("X,Y not found in cache. Generating")
    regions = pd.read_csv(regions_file, sep="\t")
    X = []
    Y = []
    for _, row in regions.iterrows():
        seq = one_hot_encode_dna(row["SEQ"])
        atac_signal = np.array(ast.literal_eval(row["ATAC"]))
        dnase_signal = np.array(ast.literal_eval(row["DNASE"]))
        feature = np.hstack((seq, atac_signal.reshape(-1, 1)))
        X.append(feature)
        Y.append(dnase_signal)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    _save_cache(X, Y, cache_dir)
    return X, Y

def create_features(
    regions: pd.DataFrame, atac_bw_file: str, fasta_file: str
) -> torch.Tensor:
    X = []
    fasta = pysam.FastaFile(fasta_file)
    atac_bw = pyBigWig.open(atac_bw_file)
    with pysam.FastaFile(fasta_file) as fasta:
        with pyBigWig.open(atac_bw_file) as atac_bw:
            for _, row in regions.iterrows():
                chrom, start, end = row[BED3_COLS]
                atac_signal = np.array(get_coverage(chrom, start, end, atac_bw))
                seq = _get_sequence(chrom, start, end, fasta)
                if not seq:
                    raise Exception(f"No sequence found for {chrom}:{start}-{end}")
                seq = one_hot_encode_dna(seq)
                feature = np.hstack((seq, atac_signal.reshape(-1, 1)))
                X.append(feature)
    return torch.tensor(np.array(X), dtype=torch.float32)

def _save_cache(X: torch.Tensor, Y: torch.Tensor, cache_dir: str) -> None:
    cache_x_file = os.path.join(cache_dir, CACHED_X_FILE)
    cache_y_file = os.path.join(cache_dir, CACHED_Y_FILE)
    torch.save(X, cache_x_file)
    torch.save(Y, cache_y_file)
    print("Saved X,Y to cache")


def _check_cache(
    regions_file: str, cache_dir: str
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    cache_x_file = os.path.join(cache_dir, CACHED_X_FILE)
    cache_y_file = os.path.join(cache_dir, CACHED_Y_FILE)
    if not os.path.exists(cache_x_file) or not os.path.exists(cache_y_file):
        return None
    
    cache_mtime = min(os.path.getmtime(cache_x_file), os.path.getmtime(cache_y_file))
    regions_mtime = os.path.getmtime(regions_file)
    if regions_mtime > cache_mtime:
        return None
    print("Loading X,Y from cache")
    X = torch.load(cache_x_file)
    Y = torch.load(cache_y_file)
    return X, Y


def get_coverage(
    chrom: str, start: int, end: int, bw: pyBigWig.pyBigWig
) -> List[float]:
    try:
        coverage = bw.values(chrom, start, end + 1)
    except Exception as e:
        print(f"Error getting coverage at {chrom}: {start}-{end}")
        return []
    coverage: List[float] = [0 if math.isnan(x) else x for x in coverage]
    return coverage


def _get_sequence(
    chrom: str, start: int, end: int, fasta: pyBigWig.pyBigWig
) -> Optional[str]:
    seq = fasta.fetch(chrom, start, end + 1)
    if not isinstance(seq, str):  # nan val
        return None
    return seq


def _split_large_regions(large_regions: pd.DataFrame, region_size: int) -> pd.DataFrame:
    new_rows = []
    for _, row in large_regions.iterrows():
        chrom = row["chrom"]
        if chrom not in NORMAL_CHROMOSOMES:
            continue
        size = row["end"] - row["start"]
        num_regions = math.ceil(size / region_size)
        for i in range(num_regions):
            start = row["start"] + (i * region_size)
            end = start + region_size
            if end > row["end"]:
                # Special care to make sure the last region split doesn't go over, but
                # is still region_size
                end = row["end"]
                start = end - region_size  # This should be in bounds
            new_row = {"chrom": chrom, "start": start, "end": end}
            new_rows.append(new_row)
    return pd.DataFrame(new_rows)


def _bedtools_sort(bed_df: pd.DataFrame, chrom_order: List[str]) -> pd.DataFrame:
    bed_df["chrom"] = pd.Categorical(
        bed_df["chrom"], categories=chrom_order, ordered=True
    )
    return bed_df.sort_values(BED3_COLS)
