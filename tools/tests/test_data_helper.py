import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

import tools.price_histories_helper as phh
import tools.trading_factors_yahoo as alpha_factors
import tools.alpha_factors_helper as afh
import tools.backtesting_functions as btf
import tools.utils as utils

"""
This is a helper module to insure all the tests have data to work with. It will generated the necessary data if 
the files do not exist. It also provides the names of the files for each data item.
"""

price_histories_test_file = 'test_data/price_hist_helper_test_data.parquet'
daily_betas_test_file = 'test_data/daily_betas_test_data.pickle'
ai_alpha_model_test_file = 'test_data/ai_alpha_model_test_data.pickle'
ai_alpha_factor_test_file = 'test_data/ai_alpha_factor_test_data.parquet'


def default_test_factors(price_histories: pd.DataFrame):
    factors_array = [
        alpha_factors.TrailingOvernightReturns(price_histories, 10).for_al(),
        alpha_factors.TrailingOvernightReturns(price_histories, 1).for_al()
    ]
    return factors_array


def get_price_histories(tickers: list = ['AAPL', 'GOOG']) -> pd.DataFrame:
    """
    Get the price histories for the test data. If the data does not exist, it will be generated.
    :return: pd.DataFrame
    """
    logger = logging.getLogger('tdh.get_price_histories')
    price_histories_path = Path(price_histories_test_file)
    logger.info(f'Getting price_histories_path: {price_histories_path}')
    symbols = utils.get_snp500().index.to_list()
    start = datetime(year=2019, month=1, day=1)
    end = datetime(year=2023, month=1, day=1)
    test_data_df = phh.from_yahoo_finance(symbols=symbols,
                                          storage_path=price_histories_path,
                                          start=start, end=end, reload=False)
    logger.info(f'Returning price_histories_test_file: {price_histories_test_file}')
    """Return only the columns for the tickers requested"""
    test_data_df = test_data_df[test_data_df.columns[test_data_df.columns.get_level_values('Symbols').isin(tickers)]]
    return test_data_df


def get_daily_betas(tickers: list) -> dict:
    """
    Get the daily betas for the test data. If the data does not exist, it will be generated.
    :return: dict
    """
    logger = logging.getLogger('tdh.get_daily_betas')
    daily_betas_path = Path(daily_betas_test_file)
    logger.info(f'Getting daily_betas_path: {daily_betas_path}')
    daily_betas = btf.generate_beta_factors(get_price_histories(tickers),
                                            storage_path=daily_betas_path, reload=False)
    logger.info(f'Returning daily_betas_path: {daily_betas_path}')
    return daily_betas


def get_alpha_vectors(tickers: list) -> pd.DataFrame:
    logger = logging.getLogger('tdh.get_alpha_vectors')

    logger.info(f'Getting alpha_vectors_path: No Stored Data')

    alpha_factors_df = get_alpha_factors(tickers)
    ai_alpha_model_path = Path(ai_alpha_model_test_file)
    model = afh.get_ai_alpha_model(alpha_factors_df, get_price_histories(tickers),
                                   n_trees=10,
                                   storage_path=ai_alpha_model_path, reload=False)
    ai_alpha_df = afh.get_ai_alpha_factor(alpha_factors_df.copy(),
                                          model,
                                          ai_alpha_name='AI_ALPHA',
                                          storage_path=None, reload=False)
    alpha_vectors_df = btf.get_alpha_vectors(ai_alpha_df, storage_path=None, reload=False)
    logger.info(f'Returning alpha_vectors')
    return alpha_vectors_df


def get_alpha_factors(tickers: list) -> pd.DataFrame:
    """
    Get the alpha factors for the test data. If the data does not exist, it will be generated.
    :return: pd.DataFrame
    """
    logger = logging.getLogger('tdh.get_alpha_factors')
    factors_df = afh.generate_factors_df(factors_array=default_test_factors(get_price_histories(tickers)))
    logger.info(f'Returning alpha_factors')
    return factors_df
