import sys
import unittest
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import tools.trading_factors_yahoo as alpha_factors
from tools.nonoverlapping_estimator import NoOverlapVoter
import tools.alpha_factors_helper as afh
import tools.price_histories_helper as phh
import tools.utils as utils
import matplotlib.pyplot as plt

logging.config.fileConfig('./test_config/logging.ini')


def default_test_factors(price_histories: pd.DataFrame):
    factors_array = [
        alpha_factors.TrailingOvernightReturns(price_histories, 10).for_al(),
        alpha_factors.TrailingOvernightReturns(price_histories, 1).for_al()
    ]
    return factors_array


class TestAlphaFactorsHelper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        symbols = ['AAPL', 'GOOG', 'AMZN', 'MMM', 'BBY', 'DUK']
        cls.snp_500_stocks = utils.get_snp500()
        start = datetime(year=2020, month=1, day=1)
        end = datetime(year=2023, month=1, day=1)
        price_histories_path = Path('test_data/alpha_factors_test_data.parquet')
        cls.price_histories = phh.from_yahoo_finance(symbols=symbols,
                                                     period='1mo',
                                                     storage_path=price_histories_path, reload=False)
        cls.close = cls.price_histories.Close
        cls.sector_helper = alpha_factors.get_sector_helper(cls.snp_500_stocks, 'GICS Sector', cls.close.columns)
        cls.factors_array = default_test_factors(cls.price_histories)
        cls.factors_df = pd.concat(cls.factors_array, axis=1)

    def test_generate_factors_df(self):
        factors_df = afh.generate_factors_df(factors_array=self.factors_array)
        self.assertIsInstance(factors_df, pd.DataFrame)
        self.assertTrue(len(factors_df) > 0)
        for factor in self.factors_array:
            self.assertIn(factor.name, factors_df.columns)
        self.assertIn('is_January', factors_df.columns)
        self.assertIn('is_December', factors_df.columns)
        self.assertIn('weekday', factors_df.columns)
        self.assertIn('quarter', factors_df.columns)
        self.assertIn('month_start', factors_df.columns)
        self.assertIn('month_end', factors_df.columns)
        self.assertIn('quarter_start', factors_df.columns)

    def test_get_alpha_factors(self):
        alpha_factors_path = Path('test_data/factors_to_use.pickle')
        alpha_factors_path.unlink(missing_ok=True)
        factors_df = afh.get_alpha_factors(factors_array=self.factors_array,
                                           storage_path=alpha_factors_path, reload=True)
        self.assertTrue(alpha_factors_path.exists())
        factors_df_reloaded = afh.get_alpha_factors(storage_path=alpha_factors_path, reload=False)
        self.assertEquals(len(factors_df), len(factors_df_reloaded))
        alpha_factors_path.unlink(missing_ok=True)

    def test_eval_factor(self):
        factor = self.factors_array[0]
        use_factor = afh.eval_factor(factor, self.price_histories)
        self.assertTrue(use_factor)

    def test_default_factors(self):
        factors_array = afh.default_factors(self.price_histories, self.sector_helper)
        self.assertIsInstance(factors_array, list)
        self.assertTrue(len(factors_array) > 0)

    def test_identify_factors_to_use(self):
        factors_to_use = afh.identify_factors_to_use(self.factors_df, self.price_histories)
        self.assertTrue(len(factors_to_use) > 0)

    def test_train_ai_alpha_model(self):
        model = afh.train_ai_alpha_model(self.factors_df, self.price_histories, n_trees=10)
        self.assertIsNotNone(model)
        self.assertTrue(np.array_equal(self.factors_df.columns, model.feature_names_in_))

    def test_get_ai_alpha_model(self):
        training_data_rows = 20
        ai_model_path = Path('test_data/alpha_ai_model.pickle')
        model = afh.get_ai_alpha_model(self.factors_df, self.price_histories,
                                       n_trees=2,
                                       storage_path=ai_model_path, reload=True)
        self.assertTrue(ai_model_path.exists())
        model_reloaded = afh.get_ai_alpha_model(self.factors_df, self.price_histories,
                                                n_trees=10,
                                                storage_path=ai_model_path, reload=False)
        self.assertTrue(np.array_equal(model.classes_, model_reloaded.classes_))
        self.assertTrue(np.array_equal(model.feature_importances_, model_reloaded.feature_importances_))
        self.assertTrue(np.array_equal(model.feature_names_in_, model_reloaded.feature_names_in_))
        self.assertEquals(model.oob_score_, model_reloaded.oob_score_)
        self.assertEquals(model.n_features_in_, model_reloaded.n_features_in_)
        ai_model_path.unlink(missing_ok=True)

    def _no_test_generate_ai_alpha(self):
        ai_alpha_model, factors_with_alpha = afh.generate_ai_alpha(self.price_histories,
                                                                   self.snp_500_stocks,
                                                                   ai_alpha_name='AI_ALPHA',
                                                                   n_trees=10,
                                                                   factors_array=self.factors_array)
        self.assertIsInstance(factors_with_alpha, pd.DataFrame)
        self.assertIsInstance(ai_alpha_model, NoOverlapVoter)

        ai_alpha = factors_with_alpha[['AI_ALPHA']].copy()
        factor_returns, _, _ = alpha_factors.evaluate_alpha(ai_alpha, self.price_histories.Close)
        cumulative_factor_returns = (1 + factor_returns).cumprod()
        total_return = cumulative_factor_returns.iloc[-1].values[0]
        print(total_return)
        plt.show()


if __name__ == '__main__':
    unittest.main()
