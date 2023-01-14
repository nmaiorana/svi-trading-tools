import datetime
import logging.config
import configparser
from platform import python_version
from pathlib import Path
import tools.utils as utils
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf


def gather_sp500_price_history():
    print(yf.__version__)
    logging.config.fileConfig('./config/logging.ini')
    logger = logging.getLogger('gather_sp500_price_history')
    logger.info(f'Python version: {python_version()}')
    logger.info(f'Pandas version: {pd.__version__}')
    logger.info(f'Pandas Data Reader version: {pdr.__version__}')

    config = configparser.ConfigParser()
    config.read('./config/config.ini')
    default_config = config["DEFAULT"]

    # Make sure we have a data directory
    Path(default_config["DataDirectory"]).mkdir(parents=True, exist_ok=True)
    price_histories_file_name = default_config["DataDirectory"] + '/' + default_config["PriceHistoriesFileName"]
    logger.info(
        f'Storing {default_config["NumberOfYearsPriceHistories"]} years of price histories in: {price_histories_file_name}')

    logger.info('Gathering S&P 500 price histories...')
    snp500_stocks = utils.get_snp500()
    stock_universe = snp500_stocks.index.to_list()
    logger.info(f'Gathered {len(stock_universe)} S&P 500 stock tickers.')

    price_histories = yf.download(tickers=stock_universe,
                                  period=default_config["NumberOfYearsPriceHistories"] + 'y',
                                  auto_adjust=True)
    price_histories.rename_axis(columns=['Attributes', 'Symbols'], inplace=True)
    price_histories = price_histories.round(2)
    # price_histories = yahoo.read()
    # yahoo.close()
    logger.info(f'PRICE_HISTORIES|Back filling from first price for each symbol...')
    price_histories.bfill(inplace=True)
    logger.info(f'PRICE_HISTORIES|Forward filling from last price for each symbol...')
    price_histories.ffill(inplace=True)
    logger.info(f'PRICE_HISTORIES|Dropping any symbol with NA...')
    price_histories.dropna(axis=1, inplace=True)
    for dropped_symbol in (set(stock_universe) - set(price_histories.columns.get_level_values(1))):
        logger.warning(f'PRICE_HISTORIES|DROPPED|{dropped_symbol}')
    logger.info(f'Read {len(price_histories)} price histories.')
    logger.info(f'PRICE_HISTORIES_FILE|{price_histories_file_name}')
    price_histories.to_csv(price_histories_file_name, index=True)
    logger.info('Price histories saved.')
    logger.info(f'PRICE_HISTORIES|{price_histories.index.min()}|{price_histories.index.max()}')
    logger.info('Done gathering S&P 500 price histories...')


print(__name__)
if __name__ == '__main__':
    gather_sp500_price_history()
