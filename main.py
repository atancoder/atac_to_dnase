import os

import click
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from atac_to_dnase.model import ATACTransformer
from atac_to_dnase.train import train_model
from atac_to_dnase.generate_bigwig import generate
from atac_to_dnase.data import get_features, get_labels, normalize_features_and_labels, normalize_features
from atac_to_dnase.utils import get_chrom_sizes
torch.manual_seed(1337)

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
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
    model = ATACTransformer(encoding_size=encoding_size, region_width=region_width, num_heads=1, num_blocks=4)
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
def save_data(training_regions, atac_bw, dnase_bw, fasta_file: str, data_folder):
    regions_df = pd.read_csv(training_regions, sep="\t")
    X, regions_df = get_features(regions_df, atac_bw, fasta_file)
    Y = get_labels(regions_df, dnase_bw)
    X, Y, stats = normalize_features_and_labels(X, Y)

    pd.DataFrame([stats]).to_csv(os.path.join(data_folder, f"{STATS_FILE}"), sep="\t", index=False)
    torch.save(X, os.path.join(data_folder, FEATURE_FILENAME))
    torch.save(Y, os.path.join(data_folder, LABELS_FILENAME))
    print(f"Saved features, labels, and normalizing params to {data_folder}")

@click.command
@click.option("--data_folder", default="data/processed")
@click.option("--saved_model", "saved_model_file", default="model.pt")
def train(data_folder, saved_model_file):
    X = torch.load(os.path.join(data_folder, FEATURE_FILENAME))
    Y = torch.load(os.path.join(data_folder, LABELS_FILENAME))
    print("Loaded data")
    # Overfit
    _, region_width, encoding_size = X.shape
    dataloader = DataLoader(TensorDataset(X, Y), BATCH_SIZE, shuffle=False)
    model = get_model(encoding_size, region_width, saved_model_file)
    train_model(model, dataloader, LEARNING_RATE, DEVICE, saved_model_file)

@click.command
@click.option("--regions", required=True)
@click.option("--atac_bw", required=True)
@click.option("--fasta", "fasta_file", type=str, required=True)
@click.option("--data_folder", default="data/processed")
@click.option("--saved_model", "saved_model_file", default="model.pt")
@click.option("--output", required=True)
def predict(regions, atac_bw, fasta_file: str, data_folder, saved_model_file, output: str):
    regions_df = pd.read_csv(regions, sep="\t")
    chrom_sizes = get_chrom_sizes(atac_bw)
    X, regions_df = get_features(regions_df, atac_bw, fasta_file)
    mean, std = pd.read_csv(os.path.join(data_folder, f"{STATS_FILE}"), sep="\t").iloc[0][["mean", "std"]]
    X = normalize_features(X, mean, std)
    _, region_width, encoding_size = X.shape
    model = get_model(encoding_size, region_width, saved_model_file)
    chrom_sizes = get_chrom_sizes(atac_bw)
    generate(regions_df, model, X, chrom_sizes, mean, std, output)

cli.add_command(save_data)
cli.add_command(train)
cli.add_command(predict)

if __name__ == "__main__":
    cli()
