import bioframe as bf
import click
import pandas as pd
import numpy as np
from utils import BED3_COLS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns



def get_crispr_positives(crispr_df):
    positives = crispr_df[crispr_df["Regulated"] == True]
    positives = positives[['chrom', 'start', 'end']].drop_duplicates()
    return positives

def get_crispr_negatives(crispr_df):
    negatives = crispr_df[crispr_df["Regulated"] != True]
    negatives = negatives[['chrom', 'start', 'end']].drop_duplicates()
    return negatives

def plot_crispr_pos(crispr_df, dhs_df, atac_df):
    crispr_pos = get_crispr_positives(crispr_df)
    crispr_dhs = bf.overlap(crispr_pos, dhs_df).fillna(0)
    crispr_atac = bf.overlap(crispr_pos, atac_df).fillna(0)
    crispr_dhs_rpm = crispr_dhs.groupby(BED3_COLS).max()["RPM_"]
    crispr_atac_rpm = crispr_atac.groupby(BED3_COLS).max()["RPM_"]
    log_fold_change = np.log2(crispr_dhs_rpm + 1) - np.log2(crispr_atac_rpm + 1)

    plt.clf()
    mean, median = np.mean(log_fold_change), np.median(log_fold_change)
    ax = sns.swarmplot(y=log_fold_change)
    ax = sns.boxplot(y=log_fold_change)
    ax.set_title("CRISPR Positives DNase to ATAC Signal")
    ax.set_xlabel("log2(dnase RPM + 1) - log2(atac RPM + 1)")
    plt.scatter([], [], label=f"n={len(log_fold_change)}\nMean={mean:.2f}\nMedian={median:.2f}")
    plt.legend()
    return ax.get_figure()

def plot_crispr_neg(crispr_df, dhs_df, atac_df):
    crispr_neg = get_crispr_negatives(crispr_df)
    crispr_dhs = bf.overlap(crispr_neg, dhs_df).fillna(0)
    crispr_atac = bf.overlap(crispr_neg, atac_df).fillna(0)
    crispr_dhs_rpm = crispr_dhs.groupby(BED3_COLS).max()["RPM_"]
    crispr_atac_rpm = crispr_atac.groupby(BED3_COLS).max()["RPM_"]
    log_fold_change = np.log(crispr_dhs_rpm + 1) - np.log(crispr_atac_rpm + 1)
    log_fold_change = log_fold_change[log_fold_change != 0].sample(1000)
    

    plt.clf()
    mean, median = np.mean(log_fold_change), np.median(log_fold_change)
    ax = sns.swarmplot(y=log_fold_change)
    ax = sns.boxplot(y=log_fold_change)
    ax.set_title("CRISPR Negatives DNase to ATAC Signal")
    ax.set_xlabel("log2(dnase RPM + 1) - log2(atac RPM + 1)")
    plt.scatter([], [], label=f"n={len(log_fold_change)}\nMean={mean:.2f}\nMedian={median:.2f}")
    plt.legend()
    return ax.get_figure()


@click.command()
@click.option("--dhs", type=str, required=True)
@click.option("--atac", type=str, required=True)
@click.option("--crispr_file", type=str, required=True)
@click.option("--output", type=str, required=True)
def main(dhs: str, atac: str, crispr_file: str, output: str):
    dhs_df = pd.read_csv(dhs, sep='\t', names=BED3_COLS + ["RPM"])
    atac_df = pd.read_csv(atac, sep='\t', names=BED3_COLS + ["RPM"])
    crispr_df = pd.read_csv(crispr_file, sep='\t')
    crispr_df.rename(columns={"chromStart": "start", "chromEnd": "end"}, inplace=True)

    with PdfPages(output) as pdf_writer:
        pdf_writer.savefig(plot_crispr_pos(crispr_df, dhs_df, atac_df))
        print("Saved positive plots")
        pdf_writer.savefig(plot_crispr_neg(crispr_df, dhs_df, atac_df))
        print("Saved negative plots")

if __name__ == "__main__":
    main()
