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
        factors = afh.generate_factors(self.test_data_df, self.sector_helper)
        self.assertTrue(len(factors) > 0)
        self.assertIn('market_dispersion_120_day', factors.columns)
        self.assertIn('is_January', factors.columns)
        self.assertIn('is_December', factors.columns)
        self.assertIn('weekday', factors.columns)
        self.assertIn('quarter', factors.columns)
        self.assertIn('month_start', factors.columns)
        self.assertIn('month_end', factors.columns)
        self.assertIn('quarter_start', factors.columns)

    def test_eval_factor(self):
        factor = alpha_factors.FactorReturns(self.test_data_df, 1).for_al()
        use_factor = afh.eval_factor(factor, self.close)
        self.assertTrue(use_factor)
        factors_df = afh.generate_factors(self.test_data_df, self.sector_helper)

        for factor_name in factors_df.columns:
            use_factor = afh.eval_factor(factors_df[factor_name], self.close)
            self.assertIsInstance(use_factor, bool)

    def test_identify_factors_to_use(self):
        factors_df = afh.generate_factors(self.test_data_df, self.sector_helper)
        factors_to_use = afh.identify_factors_to_use(factors_df, self.close)
        print(factors_to_use)
        self.assertTrue(len(factors_to_use) > 0)


if __name__ == '__main__':
    unittest.main()
