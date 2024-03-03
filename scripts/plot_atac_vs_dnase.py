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


def get_crispr_positives(crispr_df: pd.DataFrame) -> pd.DataFrame:
    positives = crispr_df[crispr_df["Regulated"] == True]
    positives = positives[["chrom", "start", "end"]].drop_duplicates()
    return positives


def get_crispr_negatives(crispr_df: pd.DataFrame) -> pd.DataFrame:
    negatives = crispr_df[crispr_df["Regulated"] != True]
    negatives = negatives[["chrom", "start", "end"]].drop_duplicates()
    return negatives


def get_figure(dnase: pd.Series, atac: pd.Series, title: str) -> Figure:
    sample_n = min(len(dnase), 500)
    dnase = dnase.sample(n=sample_n, random_state=1)
    atac = atac.sample(n=sample_n, random_state=1)
    pearson_corr = dnase.corr(atac)

    plt.clf()
    ax = sns.regplot(x=dnase,y=atac)
    ax.set_title(title)
    ax.set_xlabel("dnase counts")
    ax.set_ylabel("atac counts")
    plt.scatter(
        [], [], label=f"R={pearson_corr}"
    )
    plt.legend()
    return cast(Figure, ax.get_figure())


def plot_crispr_pos(crispr_df: pd.DataFrame, atac_df: pd.DataFrame, dnase_df: pd.DataFrame) -> Figure:
    crispr_pos = get_crispr_positives(crispr_df)
    crispr_atac_counts = bf.overlap(crispr_pos, atac_df).fillna(0)
    crispr_dnase_counts = bf.overlap(crispr_pos, dnase_df).fillna(0)

    crispr_atac_counts = crispr_atac_counts.groupby(BED3_COLS).max()["counts_"]
    crispr_dnase_counts = crispr_dnase_counts.groupby(BED3_COLS).max()["counts_"]
    return get_figure(crispr_dnase_counts, crispr_atac_counts, "CRISPR Positives DNase to ATAC Signal")


def plot_crispr_neg(crispr_df: pd.DataFrame, atac_df: pd.DataFrame, dnase_df: pd.DataFrame) -> Figure:
    crispr_neg = get_crispr_negatives(crispr_df)
    crispr_atac_counts = bf.overlap(crispr_neg, atac_df).fillna(0)
    crispr_dnase_counts = bf.overlap(crispr_neg, dnase_df).fillna(0)
    
    crispr_atac_counts = crispr_atac_counts.groupby(BED3_COLS).max()["counts_"]
    crispr_dnase_counts = crispr_dnase_counts.groupby(BED3_COLS).max()["counts_"]
    return get_figure(crispr_dnase_counts, crispr_atac_counts, "CRISPR Negatives DNase to ATAC Signal")


def plot_random_regions(atac_df: pd.DataFrame, dnase_df: pd.DataFrame) -> Figure:
    return get_figure(atac_df["counts"], dnase_df["counts"], "Random Regions DNase to ATAC Signal")


@click.command()
@click.option("--atac_bedgraph", type=str, required=True)
@click.option("--dnase_bedgraph", type=str, required=True)
@click.option("--crispr_file", type=str, required=True)
@click.option("--output_file", type=str, required=True)
def main(atac_bedgraph: str, dnase_bedgraph: str, crispr_file: str, output_file: str) -> None:
    atac_df = pd.read_csv(atac_bedgraph, sep="\t", names=BED3_COLS + ["counts"])
    dnase_df = pd.read_csv(dnase_bedgraph, sep="\t", names=BED3_COLS + ["counts"])
    crispr_df = pd.read_csv(crispr_file, sep="\t")
    crispr_df.rename(columns={"chromStart": "start", "chromEnd": "end"}, inplace=True)

    with PdfPages(output_file) as pdf_writer:
        pdf_writer.savefig(plot_crispr_pos(crispr_df, atac_df, dnase_df))
        print("Saved positive plots")
        pdf_writer.savefig(plot_crispr_neg(crispr_df, atac_df, dnase_df))
        print("Saved negative plots")
        pdf_writer.savefig(plot_random_regions(atac_df, dnase_df))
        print("Saved random plots")


if __name__ == "__main__":
    main()
