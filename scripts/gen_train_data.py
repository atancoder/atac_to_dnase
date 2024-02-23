import click
import pandas as pd
import pysam
from atac_to_dnase.utils import BED3_COLS
import math
from typing import List

REGION_SIZE = 250


def split_large_regions(large_regions: pd.DataFrame) -> pd.DataFrame:
    new_rows = []
    for _, row in large_regions.iterrows():
        size = row["end"] - row["start"]
        num_regions = math.ceil(size / REGION_SIZE)
        for i in range(num_regions):
            start = row["start"] + (i * REGION_SIZE)
            end = start + REGION_SIZE
            if end > row["end"]:
                # Special care to make sure the last region split doesn't go over, but
                # is still REGION_SIZE
                end = row["end"]
                start = end - REGION_SIZE  # This should be in bounds
            new_row = {"chrom": row["chrom"], "start": start, "end": end}
            new_rows.append(new_row)
    return pd.DataFrame(new_rows)


def bedtools_sort(bed_df: pd.DataFrame, chrom_order: List[str]):
    bed_df["chrom"] = pd.Categorical(
        bed_df["chrom"], categories=chrom_order, ordered=True
    )
    return bed_df.sort_values(BED3_COLS)


def generate_fixed_regions_df(abc_peaks_df: pd.DataFrame) -> pd.DataFrame:
    if (abc_peaks_df["end"] - abc_peaks_df["start"] < REGION_SIZE).sum() > 0:
        raise Exception(f"There are abc regions that are smaller than {REGION_SIZE}")

    large_regions = abc_peaks_df[
        abc_peaks_df["end"] - abc_peaks_df["start"] > REGION_SIZE
    ]
    split_regions = split_large_regions(large_regions)

    fixed_regions = abc_peaks_df[
        abc_peaks_df["end"] - abc_peaks_df["start"] == REGION_SIZE
    ].copy()
    fixed_regions = pd.concat([fixed_regions, split_regions], ignore_index=True)
    chrom_order = abc_peaks_df["chrom"].unique().tolist()
    fixed_regions = bedtools_sort(fixed_regions, chrom_order)
    return fixed_regions


def add_sequence(regions_df: pd.DataFrame, fasta_file: str) -> None:
    with pysam.FastaFile(fasta_file) as fasta:
        for idx, row in regions_df.iterrows():
            sequence = fasta.fetch(
                reference=row["chrom"], start=row["start"], end=row["end"] + 1
            )
            regions_df.loc[idx, "seq"] = sequence


@click.command()
@click.option("--abc_regions", type=str, required=True)
@click.option("--fasta", "fasta_file", type=str, required=True)
@click.option("--output_file", type=str, required=True)
def main(abc_regions: str, fasta_file: str, output_file: str):
    abc_df = pd.read_csv(abc_regions, sep="\t", names=BED3_COLS)
    fixed_regions = generate_fixed_regions_df(abc_df)
    add_sequence(fixed_regions, fasta_file)

    fixed_regions.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    main()
