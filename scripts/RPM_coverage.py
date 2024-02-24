import subprocess
import time
from io import StringIO

import click
import pandas as pd
import pyBigWig

from atac_to_dnase.utils import (
    BED3_COLS,
    NORMAL_CHROMOSOMES,
    estimate_bigwig_total_reads,
)


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
    regions_df: pd.DataFrame, bigwig_file: str, col_name: str
) -> pd.DataFrame:
    start_time = time.time()
    with pyBigWig.open(bigwig_file) as bw:
        regions_df = regions_df[regions_df["chrom"].isin(bw.chroms())].copy()
        total_mapped_reads = estimate_bigwig_total_reads(bw)
        for idx, row in regions_df.iterrows():
            chrom, start, end = row[BED3_COLS]
            cov = bw.stats(chrom, int(start), int(end), type="sum", exact=True)[0] or 0
            regions_df.at[idx, col_name] = 1e6 * cov / total_mapped_reads
    print(f"Time to compute {col_name}: {time.time() - start_time}")
    return regions_df


@click.command()
@click.option("--regions", type=str, required=True)
@click.option("--atac_bw", type=str, required=True)
@click.option("--dnase_bw", type=str, required=True)
@click.option("--output_file", type=str, required=True)
def main(
    regions: str,
    atac_bw: str,
    dnase_bw: str,
    output_file: str,
) -> None:
    regions_df = pd.read_csv(regions, sep="\t", usecols=range(3))
    regions_df = add_RPM_coverage(regions_df, atac_bw, "ATAC_RPM")
    regions_df = add_RPM_coverage(regions_df, dnase_bw, "DNASE_RPM")
    regions_df.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    main()
