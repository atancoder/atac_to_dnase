from typing import List

import numpy as np
import pyBigWig

BED3_COLS = ["chrom", "start", "end"]
NORMAL_CHROMOSOMES = set(["chr" + str(x) for x in range(1, 23)] + ["chrX"] + ["chrY"])
REGION_SIZE = 250
REGION_SLOP = 25


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
        mean_coverage = bw.stats(chrom, 0, length, type="mean")[0]
        total_mapped_reads += mean_coverage * length
    return total_mapped_reads
