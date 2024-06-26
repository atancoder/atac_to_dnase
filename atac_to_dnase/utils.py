from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pyBigWig

BED3_COLS = ["chrom", "start", "end"]
NORMAL_CHROMOSOMES = set(["chr" + str(x) for x in range(1, 23)] + ["chrX"] + ["chrY"])
ONE_HOT_ENCODING_SIZE = 4


def get_region_slop(region_tsv: str) -> int:
    df = pd.read_csv(region_tsv, sep="\t", nrows=1)
    return df.iloc[0]["region_slop"]


def dna_vocab_lookup(sequence: str) -> np.ndarray:
    mapping = {"A": 1, "C": 2, "G": 3, "T": 4}
    vocab_tokens = [mapping.get(nucleotide, 0) for nucleotide in sequence]
    return np.array(vocab_tokens)


def one_hot_encode_dna(sequence: str) -> np.ndarray:
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }
    zeros = [0, 0, 0, 0]
    one_hot_sequence = [mapping.get(nucleotide, zeros) for nucleotide in sequence]
    return np.array(one_hot_sequence)


def estimate_bigwig_total_reads(bw: pyBigWig.pyBigWig) -> int:
    total_mapped_reads = 0
    for chrom, length in bw.chroms().items():
        sum_coverage = bw.stats(chrom, 0, length, type="sum", exact=True)[0]
        if chrom in NORMAL_CHROMOSOMES:
            total_mapped_reads += sum_coverage
    return total_mapped_reads


def get_chrom_sizes(bw_file: str) -> Dict[str, int]:
    with pyBigWig.open(bw_file) as bw:
        return bw.chroms()
