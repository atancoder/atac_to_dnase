import os

import click
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from atac_to_dnase.model import ATACTransformer
from atac_to_dnase.train import train_model
from atac_to_dnase.generate_bigwig import generate
from atac_to_dnase.data import get_features, get_labels, normalize_features_and_labels, normalize_features
from atac_to_dnase.utils import BED3_COLS, get_chrom_sizes
from atac_to_dnase.plots import plot_losses
from typing import Optional

torch.manual_seed(1337)

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_HEADS = 1
NUM_BLOCKS = 4
EMBEDDING_MULTIPLIER = 6
FEATURE_FILENAME = "features.pt"
LABELS_FILENAME = "labels.pt"
STATS_FILE = "stats.tsv"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()  # type: ignore
    else "cpu"
)
print(f"Using {DEVICE} device")

@click.group()
def cli():
    pass

def get_model(encoding_size, region_width, saved_model_file: str) -> ATACTransformer:
    model = ATACTransformer(encoding_size=encoding_size, embedding_size=encoding_size*EMBEDDING_MULTIPLIER, region_width=region_width, num_heads=NUM_HEADS, num_blocks=NUM_BLOCKS)
    if os.path.exists(saved_model_file):
        print(f"Loading existing model: {saved_model_file}")
        model.load_state_dict(
            torch.load(saved_model_file, map_location=torch.device(DEVICE))
        )
    model.to(DEVICE)
    return model


@click.command(name="save_data")
@click.option("--training_regions", required=True)
@click.option("--atac_bw", required=True)
@click.option("--dnase_bw", required=True)
@click.option("--fasta", "fasta_file", type=str, required=True)
@click.option("--data_folder", default="data/processed")
@click.option("--output_regions", required=True)
def save_data(training_regions, atac_bw, dnase_bw, fasta_file: str, data_folder: str, output_regions:str):
    regions_df = pd.read_csv(training_regions, sep="\t")
    X, regions_df = get_features(regions_df, atac_bw, fasta_file)
    regions_df.to_csv(output_regions, sep="\t", index=False)
    Y = get_labels(regions_df, dnase_bw)
    X, Y, stats = normalize_features_and_labels(X, Y)

    pd.DataFrame([stats]).to_csv(os.path.join(data_folder, f"{STATS_FILE}"), sep="\t", index=False)
    torch.save(X, os.path.join(data_folder, FEATURE_FILENAME))
    torch.save(Y, os.path.join(data_folder, LABELS_FILENAME))
    print(f"Saved features, labels, and normalizing params to {data_folder}")

@click.command
@click.option("--data_folder", default="data/processed")
@click.option("--epochs", type=int, default="data/processed")
@click.option("--saved_model", "saved_model_file", default="model.pt")
@click.option("--loss_plot", default=None)
@click.option("--no-checkpoint", is_flag=True, default=False)
def train(data_folder: str, epochs: int, saved_model_file: str, loss_plot: Optional[str], no_checkpoint: bool):
    X = torch.load(os.path.join(data_folder, FEATURE_FILENAME))
    Y = torch.load(os.path.join(data_folder, LABELS_FILENAME))
    _, region_width, encoding_size = X.shape
    dataloader = DataLoader(TensorDataset(X, Y), BATCH_SIZE, shuffle=True)
    model = get_model(encoding_size, region_width, saved_model_file)
    losses = train_model(model, dataloader, LEARNING_RATE, DEVICE, saved_model_file, epochs=epochs, checkpoint_model=(not no_checkpoint))
    if loss_plot:
        plot_losses(losses, loss_plot)

@click.command
@click.option("--regions", required=True)
@click.option("--atac_bw", required=True)
@click.option("--fasta", "fasta_file", type=str, required=True)
@click.option("--data_folder", default="data/processed")
@click.option("--saved_model", "saved_model_file", default="model.pt")
@click.option("--output_bedgraph", default="data/processed/predictions.bedgraph")
@click.option("--output_bw", required=True)
def predict(regions, atac_bw, fasta_file: str, data_folder, saved_model_file, output_bedgraph, output_bw: str):
    regions_df = pd.read_csv(regions, sep="\t")
    chrom_sizes = get_chrom_sizes(atac_bw)
    X, regions_df = get_features(regions_df, atac_bw, fasta_file)
    stats = pd.read_csv(os.path.join(data_folder, f"{STATS_FILE}"), sep="\t").iloc[0][["mean", "std"]].to_dict()
    X = normalize_features(X, stats)

    _, region_width, encoding_size = X.shape
    model = get_model(encoding_size, region_width, saved_model_file)
    chrom_sizes = get_chrom_sizes(atac_bw)
    generate(regions_df, model, X, chrom_sizes, stats, output_bedgraph, output_bw, DEVICE)

cli.add_command(save_data)
cli.add_command(train)
cli.add_command(predict)

if __name__ == "__main__":
    cli()
