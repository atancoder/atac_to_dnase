import abc
import bioframe as bf
import click
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from typing import cast
from atac_to_dnase.utils import BED3_COLS
from atac_to_dnase.data import get_coverage
import pyBigWig

def get_crispr_positives(crispr_df: pd.DataFrame) -> pd.DataFrame:
    positives = crispr_df[crispr_df["Regulated"] == True]
    positives = positives[["chrom", "start", "end"]].drop_duplicates()
    return positives


def get_crispr_negatives(crispr_df: pd.DataFrame) -> pd.DataFrame:
    negatives = crispr_df[crispr_df["Regulated"] != True]
    negatives = negatives[["chrom", "start", "end"]].drop_duplicates()
    return negatives


def get_figure(dnase: pd.Series, atac: pd.Series, title: str) -> Figure:
    sample_n = min(len(dnase), 2000)
    dnase = dnase.sample(n=sample_n, random_state=1)
    atac = atac.sample(n=sample_n, random_state=1)
    pearson_corr = dnase.corr(atac)

    plt.clf()
    ax = sns.regplot(x=dnase,y=atac)
    ax.set_title(title)
    ax.set_xlabel("dnase counts")
    ax.set_ylabel("atac counts")
    plt.scatter(
        [], [], label=f"N={sample_n} \nR={pearson_corr}"
    )
    plt.legend()
    return cast(Figure, ax.get_figure())


def plot_crispr_pos(crispr_df: pd.DataFrame, abc_regions_df: pd.DataFrame) -> Figure:
    crispr_pos = get_crispr_positives(crispr_df)
    crispr_counts = bf.overlap(crispr_pos, abc_regions_df).fillna(0)

    crispr_atac_counts = crispr_counts.groupby(BED3_COLS).max()["atac_counts_"]
    crispr_dnase_counts = crispr_counts.groupby(BED3_COLS).max()["dnase_counts_"]
    return get_figure(crispr_dnase_counts, crispr_atac_counts, "CRISPR Positives DNase to ATAC Signal")


def plot_crispr_neg(crispr_df: pd.DataFrame, abc_regions_df: pd.DataFrame) -> Figure:
    crispr_neg = get_crispr_negatives(crispr_df)
    crispr_counts = bf.overlap(crispr_neg, abc_regions_df).fillna(0)
    
    crispr_atac_counts = crispr_counts.groupby(BED3_COLS).max()["atac_counts_"]
    crispr_dnase_counts = crispr_counts.groupby(BED3_COLS).max()["dnase_counts_"]
    return get_figure(crispr_dnase_counts, crispr_atac_counts, "CRISPR Negatives DNase to ATAC Signal")


def plot_random_regions(abc_regions_df: pd.DataFrame) -> Figure:
    return get_figure(abc_regions_df["atac_counts"], abc_regions_df["dnase_counts"], "Random Regions DNase to ATAC Signal")

def count_coverage_at_regions(abc_regions_df: pd.DataFrame, atac_bw_file: str, dnase_bw_file: str) -> pd.DataFrame:
    regions_to_skip = set()
    with pyBigWig.open(atac_bw_file) as atac_bw:
        with pyBigWig.open(dnase_bw_file) as dnase_bw:
            for idx, row in abc_regions_df.iterrows():
                idx = cast(int, idx)
                chrom, start, end = row[BED3_COLS]
                atac_signal = sum(get_coverage(chrom, start, end, atac_bw))
                dnase_signal = sum(get_coverage(chrom, start, end, dnase_bw))
                if atac_signal == 0  or dnase_signal == 0:
                    regions_to_skip.add(idx)
                    continue
                abc_regions_df.at[idx, "atac_counts"] = atac_signal
                abc_regions_df.at[idx, "dnase_counts"] = dnase_signal

    print(
        f"Skipping {len(regions_to_skip)} regions due to lack of coverage or sequence"
    )
    filtered_regions = abc_regions_df[~abc_regions_df.index.isin(regions_to_skip)]
    return abc_regions_df

@click.command()
@click.option("--abc_regions", type=str, required=True)
@click.option("--atac_bw", type=str, required=True)
@click.option("--dnase_bw", type=str, required=True)
@click.option("--crispr_file", type=str, required=True)
@click.option("--output_file", type=str, required=True)
def main(abc_regions: str, atac_bw: str, dnase_bw: str, crispr_file: str, output_file: str) -> None:
    abc_regions_df = pd.read_csv(abc_regions, sep="\t", names=BED3_COLS)
    abc_regions_df = count_coverage_at_regions(abc_regions_df, atac_bw, dnase_bw)
    crispr_df = pd.read_csv(crispr_file, sep="\t")
    crispr_df.rename(columns={"chromStart": "start", "chromEnd": "end"}, inplace=True)

    with PdfPages(output_file) as pdf_writer:
        pdf_writer.savefig(plot_crispr_pos(crispr_df, abc_regions_df))
        print("Saved positive plots")
        pdf_writer.savefig(plot_crispr_neg(crispr_df, abc_regions_df))
        print("Saved negative plots")
        pdf_writer.savefig(plot_random_regions(abc_regions_df))
        print("Saved random plots")


if __name__ == "__main__":
    main()
