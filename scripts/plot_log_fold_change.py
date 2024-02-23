import bioframe as bf
import click
import matplotlib.figure.Figure as Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

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
    return ax.get_figure()


def plot_crispr_pos(crispr_df: pd.DataFrame, rpm_df: pd.DataFrame) -> Figure:
    crispr_pos = get_crispr_positives(crispr_df)
    crispr_RPMs = bf.overlap(crispr_pos, rpm_df).fillna(0)
    crispr_dnase_rpm = crispr_RPMs.groupby(BED3_COLS).max()["DNASE_RPM_"]
    crispr_atac_rpm = crispr_RPMs.groupby(BED3_COLS).max()["ATAC_RPM_"]
    log_fold_change = np.log2(crispr_dnase_rpm + 1) - np.log2(crispr_atac_rpm + 1)

    return get_figure(log_fold_change, "CRISPR Positives DNase to ATAC Signal")


def plot_crispr_neg(crispr_df: pd.DataFrame, rpm_df: pd.DataFrame) -> Figure:
    crispr_neg = get_crispr_negatives(crispr_df)
    crispr_RPMs = bf.overlap(crispr_neg, rpm_df).fillna(0)
    crispr_dnase_rpm = crispr_RPMs.groupby(BED3_COLS).max()["DNASE_RPM_"]
    crispr_atac_rpm = crispr_RPMs.groupby(BED3_COLS).max()["ATAC_RPM_"]
    log_fold_change = np.log(crispr_dnase_rpm + 1) - np.log(crispr_atac_rpm + 1)
    log_fold_change = log_fold_change[log_fold_change != 0].sample(500)
    return get_figure(log_fold_change, "CRISPR Negatives DNase to ATAC Signal")


def plot_random_regions(rpm_df: pd.DataFrame) -> Figure:
    log_fold_change = np.log(rpm_df["DNASE_RPM"] + 1) - np.log(rpm_df["ATAC_RPM"] + 1)
    log_fold_change = log_fold_change[log_fold_change != 0].sample(500)
    return get_figure(log_fold_change, "Random Regions DNase to ATAC Signal")


@click.command()
@click.option("--rpm", type=str, required=True)
@click.option("--crispr_file", type=str, required=True)
@click.option("--output_file", type=str, required=True)
def main(rpm: str, crispr_file: str, output_file: str) -> None:
    rpm_df = pd.read_csv(rpm, sep="\t")
    if "DNASE_RPM" not in rpm_df.columns or "ATAC_RPM" not in rpm_df.columns:
        raise Exception("Must have both ATAC_RPM and DNASE_RPM columns for plotting")
    crispr_df = pd.read_csv(crispr_file, sep="\t")
    crispr_df.rename(columns={"chromStart": "start", "chromEnd": "end"}, inplace=True)

    with PdfPages(output_file) as pdf_writer:
        pdf_writer.savefig(plot_crispr_pos(crispr_df, rpm_df))
        print("Saved positive plots")
        pdf_writer.savefig(plot_crispr_neg(crispr_df, rpm_df))
        print("Saved negative plots")
        pdf_writer.savefig(plot_random_regions(rpm_df))
        print("Saved random plots")


if __name__ == "__main__":
    main()
