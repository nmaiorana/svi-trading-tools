import sys
import unittest
import logging
import pandas as pd
import os.path
from pathlib import Path
from datetime import datetime

import tools.trading_factors_yahoo as alpha_factors
from tools.nonoverlapping_estimator import NoOverlapVoter
import tools.alpha_factors_helper as afh
import tools.price_histories_helper as phh
import tools.utils as utils
import matplotlib.pyplot as plt

logging.config.fileConfig('./test_config/logging.ini')


def default_test_factors(price_histories: pd.DataFrame, sector_helper: dict):
    factors_array = [
        alpha_factors.TrailingOvernightReturns(price_histories, 10)
        .rank().zscore().smoothed(10).rank().zscore().for_al(),
        alpha_factors.FactorMeanReversion(price_histories, 5).demean(
            group_by=sector_helper.values()).rank().zscore().for_al(),
        alpha_factors.AnnualizedVolatility(price_histories, 5).rank().zscore().for_al(),
        alpha_factors.AverageDollarVolume(price_histories, 5).rank().zscore().for_al(),
        # Regime factors
        alpha_factors.MarketDispersion(price_histories, 5).for_al(),
        alpha_factors.MarketVolatility(price_histories, 5).for_al()
    ]
    return factors_array


TEST_DATA_CSV = 'test_data/alpha_factors_test_data.csv'


class TestAlphaFactorsHelper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        symbols = ['AAPL', 'GOOG', 'AMZN', 'MMM', 'BBY', 'DUK']
        cls.snp_500_stocks = utils.get_snp500()
        start = datetime(year=2020, month=1, day=1)
        end = datetime(year=2023, month=1, day=1)
        price_histories_path = Path(TEST_DATA_CSV)
        cls.test_data_df = phh.from_yahoo_finance(symbols=cls.snp_500_stocks.index.to_list(),
                                                  start=start, end=end,
                                                  storage_path=price_histories_path, reload=False)
        cls.close = cls.test_data_df.Close
        cls.snp_500_stocks = utils.get_snp500()
        cls.sector_helper = alpha_factors.get_sector_helper(cls.snp_500_stocks, 'GICS Sector', cls.close.columns)
        cls.default_test_factors = default_test_factors(cls.test_data_df, cls.sector_helper)

    def test_generate_scored_factors(self):
        factors_df = afh.generate_factors_df(factors_array=self.default_test_factors)
        self.assertTrue(len(factors_df) > 0)
        for factor in self.default_test_factors:
            self.assertIn(factor.name, factors_df.columns)
        self.assertIn('is_January', factors_df.columns)
        self.assertIn('is_December', factors_df.columns)
        self.assertIn('weekday', factors_df.columns)
        self.assertIn('quarter', factors_df.columns)
        self.assertIn('month_start', factors_df.columns)
        self.assertIn('month_end', factors_df.columns)
        self.assertIn('quarter_start', factors_df.columns)

    def test_eval_factor(self):
        factor = alpha_factors.TrailingOvernightReturns(self.test_data_df, 10).for_al()
        use_factor = afh.eval_factor(factor, self.test_data_df)
        self.assertTrue(use_factor)
        factors_df = afh.generate_factors_df(factors_array=self.default_test_factors)

        for factor_name in factors_df.columns:
            use_factor = afh.eval_factor(factors_df[factor_name], self.test_data_df)
            self.assertIsInstance(use_factor, bool)

    def test_default_factors(self):
        factors_array = afh.default_factors(self.test_data_df, self.sector_helper)
        self.assertIsInstance(factors_array, list)
        self.assertTrue(len(factors_array) > 0)

    def test_specific_factors(self):
        factors_df = afh.generate_factors_df(factors_array=self.default_test_factors)
        self.assertIsInstance(factors_df, pd.DataFrame)
        self.assertTrue(len(factors_df) > 0)
        self.assertTrue(len(factors_df) > len(self.default_test_factors))

    def test_identify_factors_to_use(self):
        factors_df = afh.generate_factors_df(factors_array=self.default_test_factors)
        factors_to_use = afh.identify_factors_to_use(factors_df, self.test_data_df)
        print(factors_to_use)
        self.assertTrue(len(factors_to_use) > 0)

    def test_train_ai_alpha_model(self):
        factors_df = afh.generate_factors_df(factors_array=self.default_test_factors)
        factors_to_use = afh.identify_factors_to_use(factors_df, self.test_data_df)
        model = afh.train_ai_alpha_model(factors_df[factors_to_use], self.test_data_df, n_trees=10)
        self.assertIsNotNone(model)

    def test_generate_ai_alpha(self):
        ai_alpha_model, factors_with_alpha = afh.generate_ai_alpha(self.test_data_df,
                                                                   self.snp_500_stocks,
                                                                   ai_alpha_name='AI_ALPHA',
                                                                   n_trees=10,
                                                                   factors_array=self.default_test_factors)
        self.assertIsInstance(factors_with_alpha, pd.DataFrame)
        self.assertIsInstance(ai_alpha_model, NoOverlapVoter)

        ai_alpha = factors_with_alpha[['AI_ALPHA']].copy()
        factor_returns, _, _ = alpha_factors.evaluate_alpha(ai_alpha, self.test_data_df.Close)
        cumulative_factor_returns = (1 + factor_returns).cumprod()
        total_return = cumulative_factor_returns.iloc[-1].values[0]
        print(total_return)
        plt.show()


if __name__ == '__main__':
    unittest.main()
