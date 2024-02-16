import math
from typing import List

import click
import pandas as pd

REGION_SIZE = 500  # Chosen b/c this is the default ABC peak size


def split_large_regions(large_regions: pd.DataFrame) -> pd.DataFrame:
    new_rows = []
    for _, row in large_regions.iterrows():
        size = row["end"] - row["start"]
        num_regions = math.ceil(size / REGION_SIZE)
        for i in range(num_regions):
            start = row["start"] + i * REGION_SIZE
            end = start + REGION_SIZE
            if end > row["end"]:
                # Special care to make sure the last region split doesn't go over, but
                # is still REGION_SIZE
                end = row["end"]
                start = end - REGION_SIZE  # This should be in bounds
            row = {"chr": row["chr"], "start": start, "end": end}
            new_rows.append(row)
    return pd.DataFrame(new_rows)


def bedtools_sort(bed_df: pd.DataFrame, chrom_order: List[str]):
    bed_df["chr"] = pd.Categorical(bed_df["chr"], categories=chrom_order, ordered=True)
    return bed_df.sort_values(["chr", "start", "end"])


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
    chrom_order = abc_peaks_df["chr"].unique().tolist()
    fixed_regions = bedtools_sort(fixed_regions, chrom_order)
    return fixed_regions


@click.command()
@click.option("--abc_regions", type=str, required=True)
@click.option("--output_file", type=str, default="fixed_regions.bed")
def main(abc_regions: str, output_file: str):
    abc_df = pd.read_csv(abc_regions, sep="\t", names=["chr", "start", "end"])
    fixed_regions = generate_fixed_regions_df(abc_df)
    fixed_regions.to_csv(output_file, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
