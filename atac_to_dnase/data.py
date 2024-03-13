import math
import os
from tkinter import NORMAL
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import pyBigWig
import pysam
import torch
from .utils import (
    BED3_COLS,
    NORMAL_CHROMOSOMES,
    one_hot_encode_dna,
    estimate_bigwig_total_reads
)

CACHED_DNA_FILE = "dna_features.pt"
CACHED_ATAC_FILE = "atac_features.pt"
CACHED_Y_FILE = "labels.pt"

class ATACSignalDataset(torch.utils.data.Dataset):  #type: ignore
    def __init__(self, X: List[Tuple[torch.Tensor]], Y: torch.Tensor):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

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


def load_features_and_labels(
    regions_file: str, atac_bw_file: str, dnase_bw_file: str, fasta_file: str, cache_dir: str, chromosomes: Set[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print("dna_X, atac_X, Y not found in cache. Generating")
    regions = pd.read_csv(regions_file, sep="\t")
    dna_X = []
    atac_X = []
    Y = []
    for chrom in chromosomes:
        result = _check_cache(regions_file, cache_dir, chrom)
        if result:
            chrom_dna_X, chrom_atac_X, chrom_Y = [r.numpy() for r in result]
        else:
            chrom_dna_X, chrom_atac_X, chrom_Y = _get_chrom_feature_and_labels(regions, chrom, atac_bw_file, dnase_bw_file, fasta_file)
            _save_cache(chrom, chrom_dna_X, chrom_atac_X, chrom_Y, cache_dir)
        dna_X.append(chrom_dna_X)
        atac_X.append(chrom_atac_X)
        Y.append(chrom_Y)

    dna_X = torch.tensor(np.concatenate(dna_X), dtype=torch.int32)
    atac_X = torch.tensor(np.concatenate(atac_X), dtype=torch.float32)
    Y = torch.tensor(np.concatenate(Y), dtype=torch.float32)
    return dna_X, atac_X, Y

def _get_chrom_feature_and_labels(regions: pd.DataFrame, chrom: str, atac_bw_file: str, dnase_bw_file: str, fasta_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dna_X = []
    atac_X = []
    Y = []
    fasta = pysam.FastaFile(fasta_file)
    atac_bw = pyBigWig.open(atac_bw_file)
    atac_total = estimate_bigwig_total_reads(atac_bw)
    dnase_bw = pyBigWig.open(dnase_bw_file)
    dnase_total = estimate_bigwig_total_reads(dnase_bw)

    relevant_regions = regions[regions["chrom"] == chrom]
    for _, row in relevant_regions.iterrows():
        chrom, start, end = row[BED3_COLS]
        seq = one_hot_encode_dna(_get_sequence(chrom, start, end, fasta))
        atac_signal = get_coverage(chrom, start, end, atac_bw) * (1e6 / atac_total)
        dnase_signal = get_coverage(chrom, start, end, dnase_bw) * (1e6 / dnase_total)
        dna_X.append(seq)
        atac_X.append(atac_signal)
        Y.append(dnase_signal)
    fasta.close()
    atac_bw.close()
    dnase_bw.close()
    return np.array(dna_X), np.array(atac_X), np.array(Y)

def create_features(
    regions: pd.DataFrame, atac_bw_file: str, fasta_file: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    dna_X = []
    atac_X = []
    fasta = pysam.FastaFile(fasta_file)
    atac_bw = pyBigWig.open(atac_bw_file)
    atac_total = estimate_bigwig_total_reads(atac_bw)
    with pysam.FastaFile(fasta_file) as fasta:
        with pyBigWig.open(atac_bw_file) as atac_bw:
            for _, row in regions.iterrows():
                chrom, start, end = row[BED3_COLS]
                atac_signal = get_coverage(chrom, start, end, atac_bw) * (1e6 / atac_total)
                seq = _get_sequence(chrom, start, end, fasta)
                if not seq:
                    raise Exception(f"No sequence found for {chrom}:{start}-{end}")
                seq = one_hot_encode_dna(seq)
                dna_X.append(seq)
                atac_X.append(atac_signal)
    dna_X = torch.tensor(np.array(dna_X), dtype=torch.int32)
    atac_X = torch.tensor(np.array(atac_X), dtype=torch.float32)
    return dna_X, atac_X

def _save_cache(chrom: str, dna_X: np.ndarray, atac_X: np.ndarray, Y: np.ndarray, cache_dir: str) -> None:
    cache_dna_file = os.path.join(cache_dir, f"{chrom}_{CACHED_DNA_FILE}")
    cache_atac_file = os.path.join(cache_dir, f"{chrom}_{CACHED_ATAC_FILE}")
    cache_y_file = os.path.join(cache_dir, CACHED_Y_FILE)
    torch.save(torch.tensor(dna_X, dtype=torch.int32), cache_dna_file)
    torch.save(torch.tensor(atac_X, dtype=torch.float32), cache_atac_file)
    torch.save(torch.tensor(Y, dtype=torch.float32), cache_y_file)
    print(f"Saved {chrom}: dna_X, atac_X, Y to cache")


def _check_cache(
    regions_file: str, cache_dir: str, chrom: str
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    cache_dna_file = os.path.join(cache_dir, f"{chrom}_{CACHED_DNA_FILE}")
    cache_atac_file = os.path.join(cache_dir, f"{chrom}_{CACHED_ATAC_FILE}")
    cache_y_file = os.path.join(cache_dir, CACHED_Y_FILE)
    if not os.path.exists(cache_dna_file) or not os.path.exists(cache_atac_file) or not os.path.exists(cache_y_file):
        return None
    cache_mtime = min(os.path.getmtime(cache_dna_file), os.path.getmtime(cache_atac_file), os.path.getmtime(cache_y_file))
    regions_mtime = os.path.getmtime(regions_file)
    if regions_mtime > cache_mtime:
        return None
    print("Loading dna_X, atac_X, Y from cache")
    dna_tensor = torch.load(cache_dna_file)
    atac_tensor = torch.load(cache_atac_file)
    label = torch.load(cache_y_file)
    return dna_tensor, atac_tensor, label


def get_coverage(
    chrom: str, start: int, end: int, bw: pyBigWig.pyBigWig
) -> np.ndarray:
    try:
        coverage = bw.values(chrom, start, end + 1)
    except Exception as e:
        print(f"Error getting coverage at {chrom}: {start}-{end}")
        return np.array([])
    coverage: List[float] = [0 if math.isnan(x) else x for x in coverage]
    return np.array(coverage)


def _get_sequence(
    chrom: str, start: int, end: int, fasta: pyBigWig.pyBigWig
) -> str:
    seq = fasta.fetch(chrom, start, end + 1)
    if not isinstance(seq, str):  # nan val
        seq_length = end - start
        return "N" * seq_length
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
