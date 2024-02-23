import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data_vectors, data_labels):
        self.data_vectors = data_vectors
        self.data_labels = data_labels

    def __len__(self):
        return len(self.data_vectors)

    def __getitem__(self, idx):
        data = self.data_vectors[idx]
        label = self.data_labels[idx]
        return data, label


# def filter_regions(regions_df):
#     """
#     Only look at data where atac and DHS signal > 0
#     """
#     return regions_df[
#         (regions_df["ATAC_RPM"] > 0) & (regions_df["DNASE_RPM"] > 0)
#     ].reset_index()


def get_dataloader(regions_tsv: str, atac_bw: str, dnase_bw: str) -> DataLoader:
    regions_df = pd.read_csv(regions_tsv, sep="\t")
    return
