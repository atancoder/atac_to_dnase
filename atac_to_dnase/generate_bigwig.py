import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, cast
from .utils import REGION_SLOP, BED3_COLS
from atac_to_dnase.data import denormalize_labels
import pyBigWig
import pandas as pd

def get_centered_predictions(y_hat:torch.Tensor) -> torch.Tensor:
    return y_hat[REGION_SLOP: -1*REGION_SLOP]

def _order_chrom_sizes(chrom_sizes: Dict[str, int], regions_df: pd.DataFrame) -> List[Tuple[str, int]]:
    ordered = []
    for chrom in regions_df['chrom'].drop_duplicates():
        ordered.append((chrom, chrom_sizes[chrom]))
    return ordered

def generate(regions_df: pd.DataFrame, model: nn.Module, X: torch.Tensor, chrom_sizes: Dict[str, int], mean: float, std: float, output_file: str) -> None:
    assert len(X) == len(regions_df), "Number of regions to predict must match ATAC signal regions"
    model.eval()
    with torch.no_grad():  # no tracking history
        with pyBigWig.open(output_file, "w") as bw:
            bw.addHeader(_order_chrom_sizes(chrom_sizes, regions_df))
            for idx, row in regions_df.iterrows():
                x = X[cast(int, idx)].unsqueeze(0)
                y_hat = model(x)
                y_hat = denormalize_labels(y_hat, mean, std)
                y_hat = get_centered_predictions(y_hat)

                chrom, start, end = row[BED3_COLS]
                bw.addEntries([chrom], [start], [end], values=[y_hat.sum().item()])
