import gzip
import subprocess
import time
from io import StringIO

import click
import pandas as pd
from utils import NORMAL_CHROMOSOMES


def count_bam_total(bam_file):
    cmd = ["samtools", "idxstat", bam_file]
    result = subprocess.check_output(cmd).decode("utf-8")
    tsv_io = StringIO(result)
    df = pd.read_csv(
        tsv_io, sep="\t", names=["chr", "size", "mapped_reads", "unmapped_reads"]
    )
    no_alt_chrom_df = df[df["chr"].isin(NORMAL_CHROMOSOMES)]
    return no_alt_chrom_df["mapped_reads"].sum()


def count_tagalign_total(tagalign_file):
    line_count = 0
    if tagalign_file.endswith(".gz"):
        open_fn = gzip.open
        read_mode = "rt"
    else:
        open_fn = open
        read_mode = "r"
    with open_fn(tagalign_file, read_mode) as f:
        for line in f:
            chrom = line.split("\t")[0]
            if chrom in NORMAL_CHROMOSOMES:
                line_count += 1
    return line_count


@click.command()
@click.option("--peak_regions", type=str, required=True)
@click.option("--profile", type=str, required=True)
@click.option("--chrom_sizes", type=str, required=True)
@click.option("--output_file", type=str, required=True)
def main(peak_regions: str, profile: str, chrom_sizes: str, output_file: str):
    start_time = time.time()
    cmd = [
        "bedtools",
        "coverage",
        "-counts",
        "-sorted",
        "-g",
        chrom_sizes,
        "-a",
        peak_regions,
        "-b",
        profile,
    ]
    result = subprocess.check_output(cmd).decode("utf-8")
    print(f"Time to count coverage: {time.time() - start_time}")
    tsv_io = StringIO(result)
    df = pd.read_csv(tsv_io, sep="\t", names=["chr", "start", "end", "counts"])

    start_time = time.time()
    if profile.endswith(".bam"):
        total_reads = count_bam_total(profile)
    elif "tagAlign" in profile:
        total_reads = count_tagalign_total(profile)
    print(f"Time to count total reads: {time.time() - start_time}")
    df["counts"] = 1e6 * df["counts"] / total_reads
    df.to_csv(output_file, sep="\t", header=False, index=False)


if __name__ == "__main__":
    main()
