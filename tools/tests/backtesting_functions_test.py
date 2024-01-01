import logging
import unittest
from pathlib import Path
import pandas as pd

import test_data_helper as tdh
import tools.alpha_factors_helper as afh
import tools.backtesting_functions as btf

logging.config.fileConfig('./test_config/logging.ini')


class BacktestingFunctions(unittest.TestCase):
    alpha_factors_path = None
    number_of_years = 5
    ai_alpha_factors_df = None

    @classmethod
    def setUpClass(cls):
        # TODO: Create required files in setup -vs- having them already there
        cls.alpha_factors_path = Path('test_data/backtest_ai_alpha_factor.parquet')
        cls.alpha_vectors_path = Path('test_data/backtest_alpha_vectors.parquet')
        cls.daily_betas_path = Path('test_data/backtest_daily_betas.pickle')
        ai_alpha_factors_df = afh.load_alpha_factors(cls.alpha_factors_path)
        symbols = ai_alpha_factors_df.index.get_level_values('Symbols').tolist()
        cls.price_histories = tdh.get_price_histories(symbols)

    def test_get_alpha_vectors(self):
        ai_alpha_factors_df = afh.load_alpha_factors(self.alpha_factors_path)
        alpha_vectors_df = btf.get_alpha_vectors(ai_alpha_factors_df)
        self.assertIsInstance(alpha_vectors_df, pd.DataFrame)
        self.assertEqual(alpha_vectors_df.index.name, 'Date')
        self.assertIsNotNone(alpha_vectors_df['AAPL'])
        alpha_vectors_path = Path('test_data/backtest_ai_alpha_vector.parquet')
        alpha_vectors_path.unlink(missing_ok=True)
        alpha_vectors_df = btf.get_alpha_vectors(ai_alpha_factors_df, alpha_vectors_path, reload=True)
        self.assertTrue(alpha_vectors_path.exists())
        alpha_vectors_df_reloaded = btf.get_alpha_vectors(self.ai_alpha_factors_df, alpha_vectors_path, reload=False)
        self.assertEqual(alpha_vectors_df.shape, alpha_vectors_df_reloaded.shape)

    def test_get_beta_factors(self):
        daily_betas = btf.generate_beta_factors(self.price_histories, 2)
        self.assertIsInstance(daily_betas, dict)

    def test_predict_optimal_holdings(self):
        alpha_vectors = btf.load_alpha_vectors(self.alpha_vectors_path)
        daily_betas = btf.load_beta_factors(self.daily_betas_path)
        opt_dates = list(daily_betas.keys())[-2:]
        optimal_holdings_df = btf.predict_optimal_holdings(alpha_vectors, daily_betas, opt_dates)
        self.assertIsInstance(optimal_holdings_df, pd.DataFrame)

    def test_backtest_factors(self):
        alpha_vectors = btf.load_alpha_vectors(self.alpha_vectors_path)
        daily_betas = tdh.get_daily_betas()
        estimated_returns_by_date_ser, optimal_holdings_df, \
            trading_costs_by_date_se = btf.backtest_factors(self.price_histories,
                                                            alpha_vectors, daily_betas, 1, 2)
        self.assertIsInstance(estimated_returns_by_date_ser, pd.Series)
        self.assertIsInstance(optimal_holdings_df, pd.DataFrame)
        self.assertIsInstance(trading_costs_by_date_se, pd.Series)


if __name__ == '__main__':
    unittest.main()
