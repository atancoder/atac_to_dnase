import os
from typing import Optional

import click
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from atac_to_dnase.data import (
    get_region_features,
    load_features_and_labels,
    split_into_fixed_region_sizes,
    create_features
)
from atac_to_dnase.generate_bigwig import generate
from atac_to_dnase.model import ATACTransformer
from atac_to_dnase.plots import plot_losses
from atac_to_dnase.train import train_model
from atac_to_dnase.utils import BED3_COLS, get_chrom_sizes, get_region_slop

torch.manual_seed(1337)

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_HEADS = 2
NUM_BLOCKS = 4
CHANNELS = 10
FEATURE_FILENAME = "features.pt"
LABELS_FILENAME = "labels.pt"
STATS_FILE = "stats.tsv"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"  # type: ignore
)
print(f"Using {DEVICE} device")


@click.group()
def cli():
    pass


def get_model(region_width, saved_model_file: Optional[str]) -> ATACTransformer:
    model = ATACTransformer(
        n_encoding=5,
        channels=CHANNELS,
        region_width=region_width,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
    )
    if saved_model_file and os.path.exists(saved_model_file):
        print(f"Loading existing model: {saved_model_file}")
        model.load_state_dict(
            torch.load(saved_model_file, map_location=torch.device(DEVICE))
        )
    model.to(DEVICE)
    return model


@click.command(name="gen_regions")
@click.option("--abc_regions", required=True)
@click.option("--region_size", type=int, required=True)
@click.option("--region_slop", type=int, required=True)
@click.option("--atac_bw", required=True)
@click.option("--dnase_bw", required=True)
@click.option("--fasta", "fasta_file", type=str, required=True)
@click.option("--output_file", required=True)
def gen_regions(
    abc_regions: str,
    region_size: int,
    region_slop: int,
    atac_bw,
    dnase_bw,
    fasta_file: str,
    output_file: str,
):
    abc_regions_df = pd.read_csv(abc_regions, sep="\t", names=BED3_COLS)
    regions = split_into_fixed_region_sizes(abc_regions_df, region_size, region_slop)
    regions = get_region_features(regions, atac_bw, dnase_bw, fasta_file)
    regions.to_csv(output_file, sep="\t", index=False)


def get_subset(X, Y):
    subset_size = 10000
    indices = torch.randperm(len(X))[:subset_size]
    return X[indices], Y[indices]


@click.command
@click.option("--regions", required=True)
@click.option("--epochs", type=int, default=100)
@click.option("--saved_model", "saved_model_file")
@click.option("--loss_plot", default=None)
@click.option("--cache_dir", default="data/cache")
def train(
    regions: str,
    epochs: int,
    saved_model_file: str,
    loss_plot: Optional[str],
    cache_dir: str,
):
    region_slop = get_region_slop(regions)
    X, Y = load_features_and_labels(regions, cache_dir)
    X, Y = get_subset(X, Y)
    dataloader = DataLoader(TensorDataset(X, Y), BATCH_SIZE, shuffle=True)
    region_width = X.shape[1]
    
    model = get_model(region_width, saved_model_file)
    losses = train_model(
        model, dataloader, LEARNING_RATE, DEVICE, saved_model_file, region_slop=region_slop, epochs=epochs
    )
    if loss_plot:
        plot_losses(losses, loss_plot)


@click.command
@click.option("--regions", required=True)
@click.option("--atac_bw", required=True)
@click.option("--fasta", "fasta_file", type=str, required=True)
@click.option("--saved_model", "saved_model_file", default="model.pt")
@click.option("--output_folder", default="results/")
def predict(
    regions,
    atac_bw,
    fasta_file: str,
    saved_model_file,
    output_folder: str,
):
    regions_df = pd.read_csv(regions, sep="\t", nrows=10)  # 10 for testing purposes
    region_slop = get_region_slop(regions)
    chrom_sizes = get_chrom_sizes(atac_bw)
    X = create_features(regions_df, atac_bw, fasta_file)
    region_width = X.shape[1]
    model = get_model(region_width, saved_model_file)
    chrom_sizes = get_chrom_sizes(atac_bw)
    generate(
        regions_df, model, X, chrom_sizes, output_folder, region_slop, DEVICE
    )


cli.add_command(gen_regions)
cli.add_command(train)
cli.add_command(predict)

if __name__ == "__main__":
    cli()
