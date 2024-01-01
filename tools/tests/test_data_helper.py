import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

import tools.price_histories_helper as phh
import tools.backtesting_functions as btf
import tools.utils as utils

"""
This is a helper module to insure all the tests have data to work with. It will generated the necessary data if 
the files do not exist. It also provides the names of the files for each data item.
"""

price_histories_test_file = 'test_data/price_hist_helper_test_data.parquet'
daily_betas_test_file = 'test_data/daily_betas_test_data.pickle'


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
    end = datetime(year=2020, month=1, day=1)
    test_data_df = phh.from_yahoo_finance(symbols=symbols,
                                          storage_path=price_histories_path,
                                          start=start, end=end, reload=False)
    logger.info(f'Returning price_histories_test_file: {price_histories_test_file}')
    """Return only the columns for the tickers requested"""
    test_data_df = test_data_df[test_data_df.columns[test_data_df.columns.get_level_values('Symbols').isin(tickers)]]
    return test_data_df


def get_daily_betas() -> dict:
    """
    Get the daily betas for the test data. If the data does not exist, it will be generated.
    :return: dict
    """
    logger = logging.getLogger('tdh.get_daily_betas')
    daily_betas_path = Path(daily_betas_test_file)
    logger.info(f'Getting daily_betas_path: {daily_betas_path}')
    daily_betas = btf.generate_beta_factors(get_price_histories(utils.get_snp500().index.to_list()), 2, storage_path=daily_betas_path, reload=False)
    logger.info(f'Returning daily_betas_path: {daily_betas_path}')
    return daily_betas
