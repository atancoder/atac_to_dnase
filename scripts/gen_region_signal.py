from typing import List

import click
import pandas as pd
import pyBigWig
import torch

from atac_to_dnase.bedgraph_interval import BedgraphInterval
from atac_to_dnase.data import get_coverage
from atac_to_dnase.utils import (
    BED3_COLS,
    NORMAL_CHROMOSOMES,
    estimate_bigwig_total_reads,
    get_region_slop,
)


def center_regions(regions_df: pd.DataFrame, region_slop: int) -> None:
    regions_df["start"] += region_slop
    regions_df["end"] -= region_slop


@click.command()
@click.option("--regions", required=True)
@click.option("--atac_bw", "atac_bw_file", type=str, required=True)
@click.option("--dnase_bw", "dnase_bw_file", type=str, required=True)
@click.option("--output_file", required=True)
def main(
    regions: str,
    atac_bw_file: str,
    dnase_bw_file: str,
    output_file: str,
):
    region_slop = get_region_slop(regions)
    regions_df = pd.read_csv(regions, sep="\t")
    del regions_df["region_slop"]
    center_regions(regions_df, region_slop)

    # Add bw signal for that region
    atac_bw = pyBigWig.open(atac_bw_file)
    atac_total = estimate_bigwig_total_reads(atac_bw)
    dnase_bw = pyBigWig.open(dnase_bw_file)
    dnase_total = estimate_bigwig_total_reads(dnase_bw)

    dnase_bedgraph = BedgraphInterval()
    atac_bedgraph = BedgraphInterval()

    for _, row in regions_df.iterrows():
        chrom, start, end = row[BED3_COLS]
        atac_signal = torch.Tensor(
            get_coverage(chrom, start, end, atac_bw) * (1e6 / atac_total)
        )
        atac_bedgraph.add_bedgraph_interval(chrom, start, end, atac_signal)

        dnase_signal = torch.Tensor(
            get_coverage(chrom, start, end, dnase_bw) * (1e6 / dnase_total)
        )
        dnase_bedgraph.add_bedgraph_interval(chrom, start, end, dnase_signal)

    atac_bedgraph_df = atac_bedgraph.to_df()
    dnase_bedgraph_df = dnase_bedgraph.to_df()
    bedgraph_df = pd.merge(
        atac_bedgraph_df.rename(columns={"count": "ATAC"}),
        dnase_bedgraph_df.rename(columns={"count": "DNASE"}),
        on=BED3_COLS,
    )
    bedgraph_df.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    main()
