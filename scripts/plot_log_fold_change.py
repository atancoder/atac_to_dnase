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


def get_figure(log_fold_change: pd.Series, title: str) -> Figure:
    plt.clf()
    mean, median = np.mean(log_fold_change), np.median(log_fold_change)
    ax = sns.swarmplot(y=log_fold_change)
    ax = sns.boxplot(y=log_fold_change)
    ax.set_title(title)
    ax.set_ylabel("log2(dnase RPM + 1) - log2(atac RPM + 1)")
    plt.scatter(
        [], [], label=f"n={len(log_fold_change)}\nMean={mean:.2f}\nMedian={median:.2f}"
    )
    plt.legend()
    return cast(Figure, ax.get_figure())


def plot_crispr_pos(crispr_df: pd.DataFrame, atac_df: pd.DataFrame, dnase_df: pd.DataFrame) -> Figure:
    crispr_pos = get_crispr_positives(crispr_df)
    crispr_atac_RPMs = bf.overlap(crispr_pos, atac_df).fillna(0)
    crispr_dnase_RPMs = bf.overlap(crispr_pos, dnase_df).fillna(0)
    crispr_dnase_rpm = crispr_atac_RPMs.groupby(BED3_COLS).max()["RPM_"]
    crispr_atac_rpm = crispr_dnase_RPMs.groupby(BED3_COLS).max()["RPM_"]
    log_fold_change = np.log2(crispr_dnase_rpm + 1) - np.log2(crispr_atac_rpm + 1)

    return get_figure(log_fold_change, "CRISPR Positives DNase to ATAC Signal")


def plot_crispr_neg(crispr_df: pd.DataFrame, atac_df: pd.DataFrame, dnase_df: pd.DataFrame) -> Figure:
    crispr_neg = get_crispr_negatives(crispr_df)
    crispr_atac_RPMs = bf.overlap(crispr_neg, atac_df).fillna(0)
    crispr_dnase_RPMs = bf.overlap(crispr_neg, dnase_df).fillna(0)
    crispr_dnase_rpm = crispr_atac_RPMs.groupby(BED3_COLS).max()["RPM_"]
    crispr_atac_rpm = crispr_dnase_RPMs.groupby(BED3_COLS).max()["RPM_"]
    log_fold_change = np.log(crispr_dnase_rpm + 1) - np.log(crispr_atac_rpm + 1)
    log_fold_change = log_fold_change[log_fold_change != 0].sample(500)
    return get_figure(log_fold_change, "CRISPR Negatives DNase to ATAC Signal")


def plot_random_regions(atac_df: pd.DataFrame, dnase_df: pd.DataFrame) -> Figure:
    log_fold_change = cast(pd.Series, np.log(atac_df["RPM"] + 1) - np.log(dnase_df["RPM"] + 1))
    log_fold_change = log_fold_change[log_fold_change != 0].sample(500)
    return get_figure(log_fold_change, "Random Regions DNase to ATAC Signal")


@click.command()
@click.option("--atac_bedgraph", type=str, required=True)
@click.option("--dnase_bedgraph", type=str, required=True)
@click.option("--crispr_file", type=str, required=True)
@click.option("--output_file", type=str, required=True)
def main(atac_bedgraph: str, dnase_bedgraph: str, crispr_file: str, output_file: str) -> None:
    atac_df = pd.read_csv(atac_bedgraph, sep="\t", names=BED3_COLS + ["RPM"])
    dnase_df = pd.read_csv(dnase_bedgraph, sep="\t", names=BED3_COLS + ["RPM"])
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
