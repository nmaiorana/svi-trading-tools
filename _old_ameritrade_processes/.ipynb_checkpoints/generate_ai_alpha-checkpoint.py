import logging
import logging.config
import configparser
from platform import python_version
from pathlib import Path
import importlib
import tools.utils as utils
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd
import pandas_datareader as pdr
from tqdm import tqdm

importlib.reload(utils)
logging.config.fileConfig('./config/logging.ini')
logger = logging.getLogger('GenerateAIAlpha')
logger.info(f'Python version: {python_version()}')
logger.info(f'Pandas version: {pd.__version__}')
logger.info(f'Pandas Data Reader version: {pdr.__version__}')

config = configparser.ConfigParser()
config.read('./config/ai_alpha_config.ini')
default_config = config["DEFAULT"]


# # Stage 1: Generate Stock Universe
# 
# - Gather stocks from specific criteria (SP500 top 50...)
# - Use stock sentiment to select stocks
# - Gather price histories

# ## Stock Universe
# 
# Here we set up the univers. This needs some work. The long term goal is to use a pipeline process to help select
# stock that are in the top 500 or something similar.
# 
# For now, we will use stocks from the portfolio, but stocks of interest (high news items), a list of well known
# stocks (this also has been augmented with some stocks that made Ameritrade's top 10 movers for a couple of days.
# This Ameritrade function has not been coded yet, but should be added down the line to automate pulling these tickers.

# # Price History data
# 
# One you have a set of investments you want to work with, you will need to pull some historical data for them.
# 
# We will obtain 5 years of price histories. In the end this will provide us with 2 years of factor data since some
#  factors are based on 1 year returns.

# Make sure we have a data directory
Path(default_config["DataDirectory"]).mkdir(parents=True, exist_ok=True) 
price_histories_file_name = default_config["DataDirectory"] + '/' + default_config["PriceHistoriesFileName"]
logger.info(f'Storing {default_config["NumberOfYearsPriceHistories"]} years of price histories in: {price_histories_file_name}')

logger.info('Gathering S&P 500 stock tickers...')
snp500_stocks = utils.get_snp500()
stock_universe = snp500_stocks.index.to_list()
logger.info(f'Gathered {len(stock_universe)} S&P 500 stock tickers.')

start = date.today() - relativedelta(years = int(default_config["NumberOfYearsPriceHistories"]))
end = date.today() - relativedelta(days = 1)
logger.info(f'Reading price histories from {start} to {end} using Yahoo Daily Reader...')
yahoo = pdr.yahoo.daily.YahooDailyReader(symbols=stock_universe, start=start, end=end, adjust_price=True, interval='d', get_actions=False, adjust_dividends=True)
price_histories = yahoo.read()
yahoo.close()
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

