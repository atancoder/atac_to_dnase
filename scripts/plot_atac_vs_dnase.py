from typing import cast

import bioframe as bf
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from atac_to_dnase.data import get_coverage
from atac_to_dnase.utils import BED3_COLS


def get_crispr_positives(crispr_df: pd.DataFrame) -> pd.DataFrame:
    positives = crispr_df[crispr_df["Regulated"] == True]
    positives = positives[["chrom", "start", "end"]].drop_duplicates()
    return positives


def get_crispr_negatives(crispr_df: pd.DataFrame) -> pd.DataFrame:
    negatives = crispr_df[crispr_df["Regulated"] != True]
    negatives = negatives[["chrom", "start", "end"]].drop_duplicates()
    return negatives


def get_scatter(dnase: pd.Series, atac: pd.Series, title: str) -> Figure:
    sample_n = min(len(dnase), 2000)
    dnase = dnase.sample(n=sample_n, random_state=1)
    atac = atac.sample(n=sample_n, random_state=1)
    pearson_corr = dnase.corr(atac)

    plt.clf()
    ax = sns.regplot(x=dnase, y=atac)
    ax.set_title(title)
    ax.set_xlabel("Dnase counts")
    ax.set_ylabel("Atac counts")
    plt.scatter([], [], label=f"N={sample_n} \nR={pearson_corr}")
    plt.plot(dnase, dnase, label="Slope = 1", color="orange")
    plt.legend()
    return cast(Figure, ax.get_figure())


def get_swarm(dnase: pd.Series, atac: pd.Series, title: str) -> Figure:
    plt.clf()
    sample_n = min(len(dnase), 500)
    dnase = dnase.sample(n=sample_n, random_state=1)
    atac = atac.sample(n=sample_n, random_state=1)
    log_fold_change = np.log2(dnase + 1) - np.log2(atac + 1)
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


def plot_crispr_pos(
    pdf_writer, crispr_df: pd.DataFrame, abc_regions_df: pd.DataFrame
) -> None:
    crispr_pos = get_crispr_positives(crispr_df)
    crispr_counts = bf.overlap(crispr_pos, abc_regions_df).fillna(0)

    crispr_atac_counts = crispr_counts.groupby(BED3_COLS).max()["atac_counts_"]
    crispr_dnase_counts = crispr_counts.groupby(BED3_COLS).max()["dnase_counts_"]

    fig = get_scatter(
        crispr_dnase_counts, crispr_atac_counts, "CRISPR Positives DNase to ATAC Signal"
    )
    pdf_writer.savefig(fig)

    fig = get_scatter(
        np.log2(crispr_dnase_counts + 1),
        np.log2(crispr_atac_counts + 1),
        "CRISPR Positives Log2 DNase to Log2 ATAC Signal",
    )
    pdf_writer.savefig(fig)

    fig = get_swarm(crispr_dnase_counts, crispr_atac_counts, "CRISPR Positives")
    pdf_writer.savefig(fig)


def plot_crispr_neg(
    pdf_writer, crispr_df: pd.DataFrame, abc_regions_df: pd.DataFrame
) -> None:
    crispr_neg = get_crispr_negatives(crispr_df)
    crispr_counts = bf.overlap(crispr_neg, abc_regions_df).fillna(0)

    crispr_atac_counts = crispr_counts.groupby(BED3_COLS).max()["atac_counts_"]
    crispr_dnase_counts = crispr_counts.groupby(BED3_COLS).max()["dnase_counts_"]

    fig = get_scatter(
        crispr_dnase_counts, crispr_atac_counts, "CRISPR Negatives DNase to ATAC Signal"
    )
    pdf_writer.savefig(fig)

    fig = get_scatter(
        np.log2(crispr_dnase_counts + 1),
        np.log2(crispr_atac_counts + 1),
        "CRISPR Negatives Log2 DNase to Log2 ATAC Signal",
    )
    pdf_writer.savefig(fig)

    fig = get_swarm(crispr_dnase_counts, crispr_atac_counts, "CRISPR Negatives")
    pdf_writer.savefig(fig)


def plot_random_regions(pdf_writer, abc_regions_df: pd.DataFrame) -> None:
    log_atac = pd.Series(np.log2(abc_regions_df["atac_counts"] + 1))
    log_dnase = pd.Series(np.log2(abc_regions_df["dnase_counts"] + 1))
    fig = get_scatter(
         log_atac,
         log_dnase,
        "Random Regions Log2 DNase to Log2 ATAC Signal",
    )
    pdf_writer.savefig(fig)
    fig = get_swarm(abc_regions_df["atac_counts"], abc_regions_df["dnase_counts"], "Random Regions")
    pdf_writer.savefig(fig)


def count_coverage_at_regions(
    abc_regions_df: pd.DataFrame, atac_bw_file: str, dnase_bw_file: str
) -> pd.DataFrame:
    regions_to_skip = set()
    with pyBigWig.open(atac_bw_file) as atac_bw:
        with pyBigWig.open(dnase_bw_file) as dnase_bw:
            from atac_to_dnase.utils import estimate_bigwig_total_reads

            atac_total = estimate_bigwig_total_reads(atac_bw) 
            dnase_total = estimate_bigwig_total_reads(dnase_bw) 

            for idx, row in abc_regions_df.iterrows():
                chrom, start, end = row[BED3_COLS]
                atac_signal = sum(get_coverage(chrom, start, end, atac_bw))
                dnase_signal = sum(get_coverage(chrom, start, end, dnase_bw))
                if atac_signal == 0 or dnase_signal == 0:
                    regions_to_skip.add(idx)
                    continue
                abc_regions_df.at[idx, "atac_counts"] = atac_signal * (1e6 / atac_total)
                abc_regions_df.at[idx, "dnase_counts"] = dnase_signal * (1e6 / dnase_total)

    print(f"Skipping {len(regions_to_skip)} regions due to lack of coverage")
    abc_regions_df = abc_regions_df[~abc_regions_df.index.isin(regions_to_skip)]
    return abc_regions_df


@click.command()
@click.option("--abc_regions", type=str, required=True)
@click.option("--atac_bw", type=str, required=True)
@click.option("--dnase_bw", type=str, required=True)
@click.option("--crispr_file", type=str, required=True)
@click.option("--output_file", type=str, required=True)
def main(
    abc_regions: str, atac_bw: str, dnase_bw: str, crispr_file: str, output_file: str
) -> None:
    abc_regions_df = pd.read_csv(abc_regions, sep="\t", names=BED3_COLS)
    abc_regions_df = count_coverage_at_regions(abc_regions_df, atac_bw, dnase_bw)
    crispr_df = pd.read_csv(crispr_file, sep="\t")
    crispr_df.rename(columns={"chromStart": "start", "chromEnd": "end"}, inplace=True)

    with PdfPages(output_file) as pdf_writer:
        plot_crispr_pos(pdf_writer, crispr_df, abc_regions_df)
        plot_crispr_neg(pdf_writer, crispr_df, abc_regions_df)
        plot_random_regions(pdf_writer, abc_regions_df)


if __name__ == "__main__":
    main()
