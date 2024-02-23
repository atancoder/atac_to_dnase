from typing import List

import numpy as np

BED3_COLS = ["chrom", "start", "end"]
NORMAL_CHROMOSOMES = set(
    ["chr" + str(x) for x in range(1, 23)] + ["chrX"] + ["chrY"] + ["chrM"]
)


def one_hot_encode_dna(sequence: List[str]) -> np.ndarray:
    # Define the mapping
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }

    # Convert each nucleotide to its one-hot representation
    one_hot_sequence = [mapping[nucleotide] for nucleotide in sequence]

    return np.array(one_hot_sequence)
