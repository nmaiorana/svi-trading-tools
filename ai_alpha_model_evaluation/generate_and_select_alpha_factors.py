import logging
import logging.config
import configparser
from platform import python_version
from pathlib import Path
import time
from datetime import datetime
import os
import pandas as pd
import numpy as np
import math
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pickle

import gather_sp500_price_histories
import tools.trading_factors_yahoo as alpha_factors
import tools.alpha_factors_helper as afh
import tools.utils as utils
import warnings

warnings.filterwarnings('ignore')

logging.config.fileConfig('../config/logging.ini')
logger = logging.getLogger('GenerateAndSelectAlphaFactors')

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20, 8)

logger.info(f'Python version: {python_version()}')
logger.info(f'Pandas version: {pd.__version__}')

config = configparser.ConfigParser()
config.read('../config/config.ini')
alpha_config = config["Alpha"]

gather_sp500_price_histories.gather_sp500_price_history(config["DEFAULT"])
price_histories_file_name = alpha_config["DataDirectory"] + '/' + alpha_config["PriceHistoriesFileName"]
logger.info(f'PRICE_HISTORIES_FILE|{price_histories_file_name}...')
price_histories = pd.read_csv(price_histories_file_name,
                              header=[0, 1], index_col=[0], parse_dates=True, low_memory=False)
logger.info(f'PRICE_HISTORIES|{price_histories.index.min()}|{price_histories.index.max()}')
logger.info(f'Using {alpha_config["NumberOfYearsForAlpha"]} years of price history data to generate alpha factors.')
latest_date = price_histories.index.max()
earliest_date = latest_date - pd.DateOffset(years=int(alpha_config["NumberOfYearsForAlpha"]))
price_histories = price_histories[(price_histories.index >= earliest_date) & (price_histories.index <= latest_date)]
logger.info(f'PRICE_HISTORIES_ALPHA|{price_histories.index.min()}|{price_histories.index.max()}')
close = price_histories.Close
logger.info(f'STOCK_TICKERS|{len(close.columns)}')
alpha_factors_file_name = alpha_config["DataDirectory"] + '/' + alpha_config["AlphaFactorsFileName"]
beta_factors_file_name = alpha_config["DataDirectory"] + '/' + alpha_config["BetaFactorsFileName"]

logger.info('Gathering stock ticker sector data...')
snp_500_stocks = utils.get_snp500()
sector_helper = alpha_factors.get_sector_helper(snp_500_stocks, 'GICS Sector', close.columns)
logger.info(f'Stock sector information gathered.')
alpha_factors_list = []
selected_factor_names = []

min_sharpe_ratio = float(alpha_config['min_sharpe_ratio'])
logger.info(f'FACTOR_EVAL|MIN_SHARPE_RATIO|{min_sharpe_ratio}')

potential_factors = [
    alpha_factors.FactorMomentum(price_histories, 252).demean(group_by=sector_helper.values()).rank().zscore(),
    alpha_factors.TrailingOvernightReturns(price_histories, 10).rank().zscore().smoothed(10).rank().zscore(),
    alpha_factors.FactorMeanReversion(price_histories, 30).demean(group_by=sector_helper.values()).rank().zscore(),
    alpha_factors.FactorMeanReversion(price_histories, 60).demean(group_by=sector_helper.values()).rank().zscore(),
    alpha_factors.FactorMeanReversion(price_histories, 90).demean(group_by=sector_helper.values()).rank().zscore(),
    alpha_factors.AnnualizedVolatility(price_histories, 20).rank().zscore(),
    alpha_factors.AnnualizedVolatility(price_histories, 120).rank().zscore(),
    alpha_factors.AverageDollarVolume(price_histories, 20).rank().zscore(),
    alpha_factors.AverageDollarVolume(price_histories, 120).rank().zscore(),
]
for potential_factor in potential_factors:
    afh.eval_factor_and_add(alpha_factors_list, selected_factor_names, potential_factor, close, min_sharpe_ratio)

# Fixed Factors
market_dispersion = alpha_factors.MarketDispersion(price_histories, 120)
alpha_factors_list.append(market_dispersion.for_al())
selected_factor_names.append(market_dispersion.factor_name)
market_volatility = alpha_factors.MarketVolatility(price_histories, 120)
alpha_factors_list.append(market_volatility.for_al())
selected_factor_names.append(market_volatility.factor_name)

for factor_created in alpha_factors_list:
    logger.info(f'CREATED_FACTOR|{factor_created.name}')

for factor_used in selected_factor_names:
    logger.info(f'USED_FACTOR|{factor_used}')
