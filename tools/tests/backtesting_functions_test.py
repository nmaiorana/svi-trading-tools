import logging
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import tools.alpha_factors_helper as afh
import tools.price_histories_helper as phh
import tools.backtesting_functions as btf

logging.config.fileConfig('./test_config/logging.ini')


class BacktestingFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ai_alpha_factors_path = Path('test_data/backtest_ai_alpha_factor.parquet')
        cls.ai_alpha_factors_df = afh.load_alpha_factors(ai_alpha_factors_path)
        price_histories_path = Path('test_data/backtest_price_histories.parquet')
        cls.number_of_years = 2
        symbols = cls.ai_alpha_factors_df.index.get_level_values('Symbols').tolist()
        cls.price_histories = phh.from_yahoo_finance(symbols=symbols,
                                                     period=str(cls.number_of_years) + 'y',
                                                     storage_path=price_histories_path, reload=False)

    def test_get_alpha_vectors(self):
        alpha_vectors_df = btf.get_alpha_vectors(self.ai_alpha_factors_df)
        self.assertIsInstance(alpha_vectors_df, pd.DataFrame)
        self.assertEquals(alpha_vectors_df.index.name, 'Date')
        self.assertIsNotNone(alpha_vectors_df['AAPL'])
        alpha_vectors_path = Path('test_data/backtest_ai_alpha_vector.parquet')
        alpha_vectors_path.unlink(missing_ok=True)
        alpha_vectors_df = btf.get_alpha_vectors(self.ai_alpha_factors_df, alpha_vectors_path, reload=True)
        self.assertTrue(alpha_vectors_path.exists())
        alpha_vectors_df_reloaded = btf.get_alpha_vectors(self.ai_alpha_factors_df, alpha_vectors_path, reload=False)
        self.assertEquals(alpha_vectors_df.shape, alpha_vectors_df_reloaded.shape)

    def test_get_beta_factors(self):
        daily_betas = btf.generate_beta_factors(self.price_histories, self.number_of_years)
        self.assertIsInstance(daily_betas, dict)


if __name__ == '__main__':
    unittest.main()
