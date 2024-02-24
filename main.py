import os

import click
import torch
from torch.utils.data import DataLoader, TensorDataset

from atac_to_dnase.data import get_dataset

BATCH_SIZE = 64
FEATURE_FILENAME = "features.pt"
LABELS_FILENAME = "labels.pt"


@click.group()
def cli():
    pass


@click.command(name="save_data")
@click.option("--training_regions", required=True)
@click.option("--atac_bw", required=True)
@click.option("--dnase_bw", required=True)
@click.option("--data_folder", default="data/processed")
def save_data(training_regions, atac_bw, dnase_bw, data_folder):
    dataset = get_dataset(training_regions, atac_bw, dnase_bw)
    X, Y = dataset.tensors
    torch.save(X, os.path.join(data_folder, FEATURE_FILENAME))
    torch.save(Y, os.path.join(data_folder, LABELS_FILENAME))
    print(f"Saved features and labels to {data_folder}")


@click.command
@click.option("--data_folder", default="data/processed")
def train(data_folder):
    X = torch.load(os.path.join(data_folder, FEATURE_FILENAME))
    Y = torch.load(os.path.join(data_folder, LABELS_FILENAME))
    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(dataset, BATCH_SIZE)
    import pdb

    pdb.set_trace()


cli.add_command(save_data)
cli.add_command(train)

if __name__ == "__main__":
    cli()
