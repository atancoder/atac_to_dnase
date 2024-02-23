import unittest

import pandas as pd
import pdb; pdb.set_trace()
from scripts.gen_train_data import REGION_SIZE, generate_fixed_regions_df
from atac_to_dnase.utils import BED3_COLS

ABC_PEAKS_REGION = "input_data/ABC_peaks.bed"


class TestCreateFixedSizeRegions(unittest.TestCase):
    def test_generate_fixed_size_regions(self) -> None:
        abc_df = pd.read_csv(ABC_PEAKS_REGION, sep="\t", names=BED3_COLS)
        fixed_regions = generate_fixed_regions_df(abc_df)

        sizes = fixed_regions["end"] - fixed_regions["start"]
        self.assertEqual(sizes.max(), REGION_SIZE)
        self.assertEqual(sizes.min(), REGION_SIZE)

        unique_regions = len(fixed_regions.drop_duplicates(BED3_COLS))
        self.assertEqual(len(fixed_regions), unique_regions)


if __name__ == "__main__":
    unittest.main()
