import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, cast
from .utils import REGION_SLOP, BED3_COLS
from atac_to_dnase.data import denormalize_labels
import pyBigWig
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd

def get_centered_predictions(Y_hat:torch.Tensor) -> torch.Tensor:
    return Y_hat[:, REGION_SLOP: -1*REGION_SLOP]

def _order_chrom_sizes(chrom_sizes: Dict[str, int], regions_df: pd.DataFrame) -> List[Tuple[str, int]]:
    ordered = []
    for chrom in regions_df['chrom'].drop_duplicates():
        ordered.append((chrom, chrom_sizes[chrom]))
    return ordered

class BedgraphInterval:
    def __init__(self) -> None:
        self._intervals: List[Dict] = []


    def add_bedgraph_interval(self, chrom: str, start: int, end: int, y_hat: torch.Tensor) -> None:
        """
        Handles merging overlaps
        We keep the predictions around for merging, but once we find out there's no overlap, we can 
        convert the predictions to a value
        """
        if self._intervals:
            last_interval = self._intervals[-1]
            num_overlaps = last_interval["end"] - start + 1
            if num_overlaps > 0:
                # Handle overlap
                first_interval_overlap_vals = last_interval["pred"][-1*num_overlaps: ]
                next_interval_overlap_vals = y_hat[:num_overlaps]
                overlap_vals = (first_interval_overlap_vals + next_interval_overlap_vals) / 2
                new_pred = torch.cat((last_interval["pred"][:-1*num_overlaps], overlap_vals, y_hat[num_overlaps:]))
                last_interval["pred"] = new_pred
                last_interval["end"] = end
            else:
                self._intervals.append({"chrom": chrom, "start": start, "end": end, "pred": y_hat})
                # cleanup last interval
                last_interval["count"] = last_interval["pred"].sum().item()
                del last_interval["pred"]
        else:
            self._intervals.append({"chrom": chrom, "start": start, "end": end, "pred": y_hat})
        
    def to_df(self) -> pd.DataFrame:
        """
        Convert the last element's predictions to a value
        """
        if self._intervals:
            last_interval = self._intervals[-1]
            if "pred" in last_interval:
                last_interval["count"] = last_interval["pred"].sum().item()
                del last_interval["pred"]
        return pd.DataFrame(self._intervals)
    

def _gen_bedgraph_intervals(regions_df: pd.DataFrame, model: nn.Module, X: torch.Tensor, mean: float, std: float) -> pd.DataFrame:
    assert len(X) == len(regions_df), "Number of regions to predict must match ATAC signal regions"
    batch_size = 64
    dataloader = DataLoader(TensorDataset(X), batch_size=batch_size)
    model.eval()
    bedgraph_interval = BedgraphInterval()
    with torch.no_grad():  # no tracking history
        for batch_id, batch in enumerate(dataloader):
            Y_hat = model(batch[0])
            Y_hat = denormalize_labels(Y_hat, mean, std)
            Y_hat = get_centered_predictions(Y_hat)
            batch_regions_df = regions_df.iloc[batch_id*batch_size: (batch_id + 1) * batch_size].reset_index()
            for idx, row in batch_regions_df.iterrows():
                chrom, start, end = row[BED3_COLS]
                y_hat = Y_hat[cast(int, idx)]
                bedgraph_interval.add_bedgraph_interval(chrom, start, end, y_hat)
            
    return bedgraph_interval.to_df()

def generate(regions_df: pd.DataFrame, model: nn.Module, X: torch.Tensor, chrom_sizes: Dict[str, int], mean: float, std: float, output_bedgraph: str, output_bw: str) -> None:
    bedgraph_intervals = _gen_bedgraph_intervals(regions_df, model, X, mean, std)
    bedgraph_intervals.to_csv(output_bedgraph, sep="\t", header=False, index=False)
    with pyBigWig.open(output_bw, "w") as bw:
        bw.addHeader(_order_chrom_sizes(chrom_sizes, regions_df))
        chroms = bedgraph_intervals["chrom"].tolist()
        starts = bedgraph_intervals["start"].tolist()
        ends = bedgraph_intervals["end"].to_list()
        vals = bedgraph_intervals["count"].to_list()
        bw.addEntries(chroms[:12], starts[:12], ends[:12], values=vals[:12])
