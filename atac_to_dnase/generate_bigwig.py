import torch
import torch.nn as nn
import os
from typing import Dict, List, Tuple, cast
from .utils import BED3_COLS
from .bedgraph_interval import BedgraphInterval
import pyBigWig
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd

def get_centered_predictions(Y_hat:torch.Tensor, region_slop: int) -> torch.Tensor:
    return Y_hat[:, region_slop: -1*region_slop]


def generate(regions_df: pd.DataFrame, model: nn.Module, X: torch.Tensor, chrom_sizes: Dict[str, int], output_folder: str, region_slop: int, device: str) -> None:
    bedgraph_intervals = _gen_bedgraph_intervals(regions_df, model, X, region_slop, device)
    bedgraph_intervals.to_csv(os.path.join(output_folder, "predictions.bedgraph"), sep="\t", header=False, index=False)
    print("Generated bedgraph file")

    with pyBigWig.open(os.path.join(output_folder, "predictions.bigWig"), "w") as bw:
        bw.addHeader(_get_ordered_chrom_sizes(chrom_sizes, regions_df))
        chroms = bedgraph_intervals["chrom"].tolist()
        starts = bedgraph_intervals["start"].tolist()
        ends = bedgraph_intervals["end"].to_list()
        vals = bedgraph_intervals["count"].to_list()
        bw.addEntries(chroms, starts, ends, values=vals)
    print("Generated bigWig file")


def _get_ordered_chrom_sizes(chrom_sizes: Dict[str, int], regions_df: pd.DataFrame) -> List[Tuple[str, int]]:
    ordered = []
    for chrom in regions_df['chrom'].drop_duplicates():
        ordered.append((chrom, chrom_sizes[chrom]))
    return ordered
    

def _gen_bedgraph_intervals(regions_df: pd.DataFrame, model: nn.Module, X: torch.Tensor, region_slop: int, device:str) -> pd.DataFrame:
    assert len(X) == len(regions_df), "Number of regions to predict must match ATAC signal regions"
    batch_size = 64
    dataloader = DataLoader(TensorDataset(X), batch_size=batch_size)
    bedgraph_interval = BedgraphInterval()
    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            Y_hat = model(batch[0].to(device))
            Y_hat = get_centered_predictions(Y_hat, region_slop)
            batch_regions_df = regions_df.iloc[batch_id*batch_size: (batch_id + 1) * batch_size].reset_index(drop=True)
            for idx, row in batch_regions_df.iterrows():
                chrom, start, end = row[BED3_COLS]
                y_hat = Y_hat[cast(int, idx)]
                bedgraph_interval.add_bedgraph_interval(chrom, start, end, y_hat)
        
    return bedgraph_interval.to_df()