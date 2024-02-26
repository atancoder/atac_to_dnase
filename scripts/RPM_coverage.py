import subprocess
import time
from io import StringIO
import torch

import click
import pandas as pd
import pyBigWig

from atac_to_dnase.utils import (
    BED3_COLS,
    NORMAL_CHROMOSOMES,
    estimate_bigwig_total_reads,
    compute_RPM
)
from atac_to_dnase.bedgraph_interval import BedgraphInterval


def count_bam_total(bam_file: str) -> int:
    cmd = ["samtools", "idxstat", bam_file]
    result = subprocess.check_output(cmd).decode("utf-8")
    tsv_io = StringIO(result)
    df = pd.read_csv(
        tsv_io, sep="\t", names=["chr", "size", "mapped_reads", "unmapped_reads"]
    )
    no_alt_chrom_df = df[df["chr"].isin(NORMAL_CHROMOSOMES)]
    return no_alt_chrom_df["mapped_reads"].sum()


def add_RPM_coverage(
    regions_df: pd.DataFrame, bigwig_file: str
) -> pd.DataFrame:
    start_time = time.time()
    bedgraph_interval = BedgraphInterval()
    with pyBigWig.open(bigwig_file) as bw:
        regions_df = regions_df[regions_df["chrom"].isin(bw.chroms())].copy()
        total_mapped_reads = estimate_bigwig_total_reads(bw)
        for _, row in regions_df.iterrows():
            chrom, start, end = row[BED3_COLS]
            bp_cov = torch.Tensor(bw.values(chrom, start, end+1))
            rpm_bp_cov = torch.Tensor(compute_RPM(bp_cov.numpy(), total_mapped_reads))
            bedgraph_interval.add_bedgraph_interval(chrom, start, end, rpm_bp_cov)
    print(f"Time to compute {bigwig_file}: {time.time() - start_time}")
    return bedgraph_interval.to_df()


@click.command()
@click.option("--regions", type=str, required=True)
@click.option("--atac_bw", type=str, required=True)
@click.option("--dnase_bw", type=str, required=True)
@click.option("--output_atac", type=str, required=True)
@click.option("--output_dnase", type=str, required=True)
def main(
    regions: str,
    atac_bw: str,
    dnase_bw: str,
    output_atac: str,
    output_dnase: str,
) -> None:
    regions_df = pd.read_csv(regions, sep="\t", usecols=range(3))
    atac_bedgraph_df = add_RPM_coverage(regions_df, atac_bw)
    dnase_bedgraph_df = add_RPM_coverage(regions_df, dnase_bw)

    atac_bedgraph_df.to_csv(output_atac, sep="\t", index=False, header=False)
    dnase_bedgraph_df.to_csv(output_dnase, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
