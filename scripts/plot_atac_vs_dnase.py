from typing import Optional, cast

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
from atac_to_dnase.utils import BED3_COLS, estimate_bigwig_total_reads


def get_crispr_positives(crispr_df: pd.DataFrame) -> pd.DataFrame:
    positives = crispr_df[crispr_df["Regulated"] == True]
    positives = positives[["chrom", "start", "end"]].drop_duplicates()
    return positives


def get_crispr_positives_NR(crispr_df: pd.DataFrame) -> pd.DataFrame:
    NR = "/oak/stanford/groups/engreitz/Users/atan5133/data/scATAC/bins/fragment_NR_abc.bedgraph"
    df = pd.read_csv(NR, sep="\t", names=BED3_COLS + ["counts"], usecols=range(4))
    positives = get_crispr_positives(crispr_df)
    filtered_positives = bf.overlap(positives, df, how="inner")
    filtered_positives = filtered_positives[filtered_positives["counts_"] > 0]
    top_quantile = min(len(positives) // 20, len(filtered_positives))
    return filtered_positives.sort_values("counts_", ascending=False)[:top_quantile][
        BED3_COLS
    ]


def get_crispr_positives_NFR(crispr_df: pd.DataFrame) -> pd.DataFrame:
    NFR = "/oak/stanford/groups/engreitz/Users/atan5133/data/scATAC/bins/fragment_NFR_abc.bedgraph"
    df = pd.read_csv(NFR, sep="\t", names=BED3_COLS + ["counts"], usecols=range(4))
    positives = get_crispr_positives(crispr_df)
    filtered_positives = bf.overlap(positives, df, how="inner")
    filtered_positives = filtered_positives[filtered_positives["counts_"] > 0]
    top_quantile = min(len(positives) // 20, len(filtered_positives))
    return filtered_positives.sort_values("counts_", ascending=False)[:top_quantile][
        BED3_COLS
    ]


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


def plot_difference_w_abc_overlaps(
    pdf_writer,
    regions: pd.DataFrame,
    abc_region_coverage: pd.DataFrame,
    region_type: str,
):
    crispr_counts = bf.overlap(regions, abc_region_coverage).fillna(0)

    crispr_atac_counts = crispr_counts.groupby(BED3_COLS).max()["atac_counts_"]
    crispr_dnase_counts = crispr_counts.groupby(BED3_COLS).max()["dnase_counts_"]
    plot_dnase_vs_atac(pdf_writer, crispr_dnase_counts, crispr_atac_counts, region_type)


def plot_dnase_vs_atac(pdf_writer, dnase_counts, atac_counts, region_type):
    fig = get_scatter(dnase_counts, atac_counts, f"{region_type} DNase to ATAC Signal")
    pdf_writer.savefig(fig)
    fig = get_scatter(
        np.log2(dnase_counts + 1),
        np.log2(atac_counts + 1),
        f"{region_type} Log2 DNase to Log2 ATAC Signal",
    )
    pdf_writer.savefig(fig)

    fig = get_swarm(dnase_counts, atac_counts, f"{region_type}")
    pdf_writer.savefig(fig)


def plot_difference(
    pdf_writer,
    regions: pd.DataFrame,
    dnase_bw_file: str,
    atac_bw_file: str,
    region_type: str,
):
    region_coverage = count_coverage_at_regions(regions, dnase_bw_file, atac_bw_file)
    plot_dnase_vs_atac(
        pdf_writer,
        region_coverage["dnase_counts"],
        region_coverage["atac_counts"],
        region_type,
    )


def plot_random_regions(pdf_writer, coverage_df: pd.DataFrame) -> None:
    log_atac = pd.Series(np.log2(coverage_df["atac_counts"] + 1))
    log_dnase = pd.Series(np.log2(coverage_df["dnase_counts"] + 1))
    fig = get_scatter(
        log_atac,
        log_dnase,
        "Random Regions Log2 DNase to Log2 ATAC Signal",
    )
    pdf_writer.savefig(fig)
    fig = get_swarm(
        coverage_df["atac_counts"], coverage_df["dnase_counts"], "Random Regions"
    )
    pdf_writer.savefig(fig)


def count_coverage_at_regions(
    coverage_df: pd.DataFrame,
    dnase_bw_file: str,
    atac_bw_file: str,
) -> pd.DataFrame:
    regions_to_skip = set()
    with pyBigWig.open(atac_bw_file) as atac_bw:
        with pyBigWig.open(dnase_bw_file) as dnase_bw:
            atac_total = estimate_bigwig_total_reads(atac_bw)
            dnase_total = estimate_bigwig_total_reads(dnase_bw)

            for idx, row in coverage_df.iterrows():
                chrom, start, end = row[BED3_COLS]
                atac_signal = get_coverage(chrom, start, end, atac_bw).sum()
                dnase_signal = get_coverage(chrom, start, end, dnase_bw).sum()
                if atac_signal == 0 or dnase_signal == 0:
                    regions_to_skip.add(idx)
                    continue
                coverage_df.at[idx, "atac_counts"] = atac_signal * (1e6 / atac_total)
                coverage_df.at[idx, "dnase_counts"] = dnase_signal * (1e6 / dnase_total)

    print(f"Skipping {len(regions_to_skip)} regions due to lack of coverage")
    coverage_df = coverage_df[~coverage_df.index.isin(regions_to_skip)]
    return coverage_df


@click.command()
@click.option(
    "--abc_regions",
    type=str,
    help="If you wish to get coverage at ABC peaks to overlap with CRISPR",
)
@click.option("--atac_bw", type=str, required=True)
@click.option("--dnase_bw", type=str, required=True)
@click.option("--crispr_file", type=str, required=True)
@click.option(
    "--bed_file", type=str, help="BED file of regions you wish to compare signal at"
)
@click.option("--output_file", type=str, required=True)
def main(
    abc_regions: Optional[str],
    atac_bw: str,
    dnase_bw: str,
    crispr_file: str,
    bed_file: Optional[str],
    output_file: str,
) -> None:
    crispr_df = pd.read_csv(crispr_file, sep="\t")
    crispr_df.rename(columns={"chromStart": "start", "chromEnd": "end"}, inplace=True)

    with PdfPages(output_file) as pdf_writer:
        if abc_regions:
            abc_regions_df = pd.read_csv(abc_regions, sep="\t", names=BED3_COLS)
            abc_coverage_df = count_coverage_at_regions(
                abc_regions_df, atac_bw, dnase_bw
            )
            plot_difference_w_abc_overlaps(
                pdf_writer,
                get_crispr_positives(crispr_df),
                abc_coverage_df,
                region_type="CRISPR Positives",
            )
            plot_difference_w_abc_overlaps(
                pdf_writer,
                get_crispr_negatives(crispr_df),
                abc_coverage_df,
                region_type="CRISPR Negatives",
            )
            plot_random_regions(pdf_writer, abc_coverage_df)
        else:
            plot_difference(
                pdf_writer,
                get_crispr_positives(crispr_df),
                dnase_bw,
                atac_bw,
                region_type="CRISPR Positives",
            )
            plot_difference(
                pdf_writer,
                get_crispr_positives_NR(crispr_df),
                dnase_bw,
                atac_bw,
                region_type="CRISPR Positives",
            )
            plot_difference(
                pdf_writer,
                get_crispr_positives_NFR(crispr_df),
                dnase_bw,
                atac_bw,
                region_type="CRISPR Positives",
            )
            plot_difference(
                pdf_writer,
                get_crispr_negatives(crispr_df),
                dnase_bw,
                atac_bw,
                region_type="CRISPR Negatives",
            )

        if bed_file:
            region_df = pd.read_csv(
                bed_file, sep="\t", names=BED3_COLS, usecols=range(3)
            )
            region_df = region_df.sample(min(len(region_df), 2000))
            plot_difference(
                pdf_writer,
                region_df,
                dnase_bw,
                atac_bw,
                region_type="Custom BED",
            )


if __name__ == "__main__":
    main()
