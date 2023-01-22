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

import tools.price_histories_helper as phh
import tools.trading_factors_yahoo as alpha_factors
import tools.alpha_factors_helper as afh
import tools.configuration_helper as config_helper
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
alpha_config = config["AIAlpha"]

snp_500_stocks = utils.get_snp500()
price_histories = phh.from_yahoo_finance(symbols=snp_500_stocks.index.to_list(),
                                         period=config_helper.get_number_of_years_of_price_histories(alpha_config),
                                         storage_path=config_helper.get_price_histories_path(alpha_config),
                                         reload=False)
ai_alpha_name = alpha_config['AIAlphaName']
# TODO: Check for Alpha Factors file, if exists skip making one
# TODO: Check for Alpha model file, if exists skip making one
# TODO: Check for Alpha vectors file, if exists skip making one
# This will most likely involve breaking up the following call
ai_alpha_model, factors_with_alpha = afh.generate_ai_alpha(price_histories,
                                                           snp_500_stocks,
                                                           ai_alpha_name,
                                                           float(alpha_config['min_sharpe_ratio']),
                                                           int(alpha_config['ForwardPredictionDays']),
                                                           int(alpha_config['PredictionQuantiles']),
                                                           int(alpha_config['RandomForestNTrees']))

ai_alpha = factors_with_alpha[[ai_alpha_name]].copy()
factor_returns, _, _ = alpha_factors.evaluate_alpha(ai_alpha, price_histories.Close)
cumulative_factor_returns = (1 + factor_returns).cumprod()
total_return = cumulative_factor_returns.iloc[-1].values[0]
logger.info(f'Total return on {ai_alpha_name} is {total_return}')

plt.show()

ai_alpha = factors_with_alpha[ai_alpha_name].copy()
alpha_vector = ai_alpha.reset_index().pivot(index='Date', columns='Symbols', values=ai_alpha_name)

# TODO: Save Factors
# TODO: Create Configuration for factors_to_use for model training features
# TODO: Train model on factors_to_use
# TODO: Score model
# TODO: Backtest
# TODO: Evaluate and push to prod for use
