import os
from typing import Optional

import click
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

from atac_to_dnase.data import (
    get_region_features,
    load_features_and_labels,
    split_into_fixed_region_sizes,
    create_features
)
from atac_to_dnase.generate_bigwig import generate
from atac_to_dnase.model import ATACTransformer
from atac_to_dnase.plots import plot_losses
from atac_to_dnase.train import train_model, evaluate_model
from atac_to_dnase.utils import BED3_COLS, get_chrom_sizes, get_region_slop

torch.manual_seed(1337)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"  # type: ignore
)
print(f"Using {DEVICE} device")


# Hyperparams
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_HEADS = 2
NUM_BLOCKS = 4
CHANNELS = 10

# LR grid search
LEARNING_RATES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


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


def get_subset(X, Y, subset_size: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
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

@click.command(name="lr_grid_search")
@click.option("--regions", required=True)
@click.option("--epochs", type=int, default=5)
@click.option("--plots_dir", default="plots")
@click.option("--cache_dir", default="data/cache")
def lr_grid_search(
    regions: str,
    epochs: int,
    plots_dir: str,
    cache_dir: str,
):
    region_slop = get_region_slop(regions)
    X, Y = load_features_and_labels(regions, cache_dir)
    X, Y = get_subset(X, Y, subset_size=10000)
    train_size = int(10000 * .8)
    train_X, train_Y = X[:train_size], Y[:train_size]
    val_X, val_Y = X[train_size:], Y[train_size:]
    train_dataloader = DataLoader(TensorDataset(train_X, train_Y), BATCH_SIZE, shuffle=False)
    val_dataloader = DataLoader(TensorDataset(val_X, val_Y), BATCH_SIZE, shuffle=False)
    region_width = X.shape[1]

    best_lr = None
    lowest_val_loss = float('inf')
    for lr in LEARNING_RATES:
        model = get_model(region_width, None)
        print(f"Training model with LR: {lr}")
        losses = train_model(
            model, train_dataloader, LEARNING_RATE, DEVICE, saved_model_file=None, region_slop=region_slop, epochs=epochs
        )
        plot_losses(losses, output_file=os.path.join(plots_dir, f"lr_{lr}_plot.pdf"))
        print("Evaluating model")
        val_loss = evaluate_model(model, val_dataloader, DEVICE, region_slop)
        print(f"{lr} has validation loss of {val_loss}")
        if val_loss < lowest_val_loss:
            best_lr = lr
            lowest_val_loss = val_loss
    print(f"Best Learning Rate: {best_lr} with Validation Loss: {lowest_val_loss}")

cli.add_command(gen_regions)
cli.add_command(train)
cli.add_command(predict)
cli.add_command(lr_grid_search)

if __name__ == "__main__":
    cli()
