import os
import sys
import unittest

import pandas as pd

SCRIPTS_DIR = os.path.abspath("workflow/scripts")
sys.path.insert(0, SCRIPTS_DIR)
from create_fixed_size_regions import REGION_SIZE, generate_fixed_regions_df
from utils import BED3_COLS

ABC_PEAKS_REGION = "data/ABC_peaks.bed"



class TestCreateFixedSizeRegions(unittest.TestCase):
    def test_create_fixed_size_regions(self) -> None:
        abc_df = pd.read_csv(ABC_PEAKS_REGION, sep="\t", names=BED3_COLS)
        fixed_regions = generate_fixed_regions_df(abc_df)

        sizes = fixed_regions["end"] - fixed_regions["start"]
        self.assertEqual(sizes.max(), REGION_SIZE)
        self.assertEqual(sizes.min(), REGION_SIZE)

        unique_regions = len(fixed_regions.drop_duplicates(BED3_COLS))
        self.assertEqual(len(fixed_regions), unique_regions)


if __name__ == "__main__":
    unittest.main()
