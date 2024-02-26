import torch
import torch.nn as nn
from typing import Dict, List, Tuple, cast
from .utils import REGION_SLOP, BED3_COLS
from .bedgraph_interval import BedgraphInterval
from atac_to_dnase.data import denormalize_labels
import pyBigWig
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd

def get_centered_predictions(Y_hat:torch.Tensor) -> torch.Tensor:
    return Y_hat[:, REGION_SLOP: -1*REGION_SLOP]


def generate(regions_df: pd.DataFrame, model: nn.Module, X: torch.Tensor, chrom_sizes: Dict[str, int], stats: Dict[str, float], output_bedgraph: str, output_bw: str, device: str) -> None:
    bedgraph_intervals = _gen_bedgraph_intervals(regions_df, model, X, stats, device)
    bedgraph_intervals.to_csv(output_bedgraph, sep="\t", header=False, index=False)
    with pyBigWig.open(output_bw, "w") as bw:
        bw.addHeader(_order_chrom_sizes(chrom_sizes, regions_df))
        chroms = bedgraph_intervals["chrom"].tolist()
        starts = bedgraph_intervals["start"].tolist()
        ends = bedgraph_intervals["end"].to_list()
        vals = bedgraph_intervals["count"].to_list()
        bw.addEntries(chroms, starts, ends, values=vals)


def _order_chrom_sizes(chrom_sizes: Dict[str, int], regions_df: pd.DataFrame) -> List[Tuple[str, int]]:
    ordered = []
    for chrom in regions_df['chrom'].drop_duplicates():
        ordered.append((chrom, chrom_sizes[chrom]))
    return ordered
    

def _gen_bedgraph_intervals(regions_df: pd.DataFrame, model: nn.Module, X: torch.Tensor, stats: Dict[str, float], device:str) -> pd.DataFrame:
    assert len(X) == len(regions_df), "Number of regions to predict must match ATAC signal regions"
    batch_size = 64
    dataloader = DataLoader(TensorDataset(X), batch_size=batch_size)
    model.eval()
    bedgraph_interval = BedgraphInterval()
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            Y_hat = model(batch[0].to(device))
            Y_hat = get_centered_predictions(Y_hat)
            Y_hat = denormalize_labels(Y_hat, stats)
            batch_regions_df = regions_df.iloc[batch_id*batch_size: (batch_id + 1) * batch_size].reset_index(drop=True)
            for idx, row in batch_regions_df.iterrows():
                chrom, start, end = row[BED3_COLS]
                y_hat = Y_hat[cast(int, idx)]
                bedgraph_interval.add_bedgraph_interval(chrom, start, end, y_hat)
        
    return bedgraph_interval.to_df()