import unittest
import logging
import pandas as pd
import os.path
from pathlib import Path
from datetime import datetime

import price_histories_helper
import tools.trading_factors_yahoo as alpha_factors
import tools.alpha_factors_helper as afh
import tools.price_histories_helper as phh


class TestAlphaFactorsHelper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start = datetime(year=2019, month=1, day=1)
        end = datetime(year=2020, month=1, day=1)
        cls.test_data_df = phh.from_yahoo_finance(symbols=['AAPL', 'GOOG'],
                                                  storage_path=Path('test_data/test_data.csv'),
                                                  start=start, end=end)
        cls.close = cls.test_data_df.Close
        cls.snp_500_stocks = phh.load_snp500_symbols(Path('./test_data/snp500.csv'))
        cls.sector_helper = alpha_factors.get_sector_helper(cls.snp_500_stocks, 'GICS Sector', cls.close.columns)

    def test_generate_scored_factors(self):
        factors = afh.generate_scored_factors(self.test_data_df, self.sector_helper)
        self.assertTrue(len(factors) > 0)

    def test_generate_fixed_factors(self):
        factors = afh.generate_fixed_factors(self.test_data_df)
        self.assertTrue(len(factors) > 0)

    def test_eval_factor(self):
        factor = alpha_factors.FactorReturns(self.test_data_df, 1).for_al()
        use_factor = afh.eval_factor(factor, self.close)
        self.assertTrue(use_factor)
        factors = afh.generate_scored_factors(self.test_data_df, self.sector_helper)

        for scored_factor in factors:
            use_factor = afh.eval_factor(scored_factor, self.close)
            self.assertIsInstance(use_factor, bool)

        factors = afh.generate_fixed_factors(self.test_data_df)

        for scored_factor in factors:
            use_factor = afh.eval_factor(scored_factor, self.close)
            self.assertIsInstance(use_factor, bool)


if __name__ == '__main__':
    unittest.main()
