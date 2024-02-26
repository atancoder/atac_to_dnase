from typing import List, Dict
import torch
import pandas as pd

class BedgraphInterval:
    def __init__(self) -> None:
        self._intervals: List[Dict] = []


    def add_bedgraph_interval(self, chrom: str, start: int, end: int, bp_coverage: torch.Tensor) -> None:
        """
        Handles merging overlaps
        We keep the bp coverage around for merging, but once we find out there's no overlap, we can 
        convert the bp coverage to a value
        """
        if self._intervals:
            last_interval = self._intervals[-1]
            num_overlaps = last_interval["end"] - start + 1
            if num_overlaps > 0:
                # Handle overlap
                first_interval_overlap_vals = last_interval["bp_coverage"][-1*num_overlaps: ]
                next_interval_overlap_vals = bp_coverage[:num_overlaps]
                overlap_vals = (first_interval_overlap_vals + next_interval_overlap_vals) / 2
                new_pred = torch.cat((last_interval["bp_coverage"][:-1*num_overlaps], overlap_vals, bp_coverage[num_overlaps:]))
                last_interval["bp_coverage"] = new_pred
                last_interval["end"] = end
            else:
                self._intervals.append({"chrom": chrom, "start": start, "end": end, "bp_coverage": bp_coverage})
                # cleanup last interval
                last_interval["count"] = max(last_interval["bp_coverage"].sum().item(), 0)
                del last_interval["bp_coverage"]
        else:
            self._intervals.append({"chrom": chrom, "start": start, "end": end, "bp_coverage": bp_coverage})
        
    def to_df(self) -> pd.DataFrame:
        """
        Convert the last element's predictions to a value
        """
        if self._intervals:
            last_interval = self._intervals[-1]
            if "bp_coverage" in last_interval:
                last_interval["count"] = max(last_interval["bp_coverage"].sum().item(), 0)
                del last_interval["bp_coverage"]
        return pd.DataFrame(self._intervals)