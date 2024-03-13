from typing import List
import pyBigWig
import click
import pandas as pd

from atac_to_dnase.utils import BED3_COLS, NORMAL_CHROMOSOMES, get_region_slop, estimate_bigwig_total_reads
from atac_to_dnase.data import get_coverage

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

    for idx, row in regions_df.iterrows():
        chrom, start, end = row[BED3_COLS]
        atac_signal = get_coverage(chrom, start, end, atac_bw).sum() * (1e6 / atac_total)
        dnase_signal = get_coverage(chrom, start, end, dnase_bw).sum() * (1e6 / dnase_total)
        regions_df.at[idx, "ATAC"] = atac_signal
        regions_df.at[idx, "DNASE"] = dnase_signal
    
    regions_df.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    main()
