from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyBigWig
import torch
from torch.utils.data import DataLoader, TensorDataset

from .utils import BED3_COLS, estimate_bigwig_total_reads, one_hot_encode_dna


def filter_regions(regions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Only look at data where atac and DHS signal > 0
    """
    return regions_df[~regions_df["seq"].isna()].reset_index()


def gen_features_and_label(
    chrom: str,
    start: int,
    end: int,
    seq: str,
    atac_bw: pyBigWig.pyBigWig,
    atac_total_reads: int,
    dnase_bw: pyBigWig.pyBigWig,
    dnase_total_reads: int,
) -> Optional[Tuple[List[List[float]], List[float]]]:
    ohe = one_hot_encode_dna(seq)
    atac_signal: List[float] = atac_bw.values(chrom, start, end + 1)
    dnase_signal: List[float] = dnase_bw.values(chrom, start, end + 1)
    if sum(atac_signal) == 0 or sum(dnase_signal) == 0:
        # Only find data where we have signal in both
        return None

    combined_features = [
        ohe_vector + [atac_signal_val]
        for ohe_vector, atac_signal_val in zip(ohe, atac_signal)
    ]

    return combined_features, dnase_signal


def get_dataset(
    regions_tsv: str, atac_bw_file: str, dnase_bw_file: str
) -> TensorDataset:
    regions_df = filter_regions(pd.read_csv(regions_tsv, sep="\t"))
    X, Y = [], []
    regions_skipped = 0
    with pyBigWig.open(atac_bw_file) as atac_bw:
        with pyBigWig.open(dnase_bw_file) as dnase_bw:
            atac_total_reads = estimate_bigwig_total_reads(atac_bw)
            dnase_total_reads = estimate_bigwig_total_reads(dnase_bw)
            for idx, row in regions_df.iterrows():
                chrom, start, end = row[BED3_COLS]
                result = gen_features_and_label(
                    chrom,
                    start,
                    end,
                    row["seq"],
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
    print(f"Skipping {regions_skipped} regions due to lack of coverage")
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    return TensorDataset(X, Y)


def get_dataloader(
    dataset: TensorDataset, atac_bw_file: str, dnase_bw_file: str, batch_size: int
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size)
