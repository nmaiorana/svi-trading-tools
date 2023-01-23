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
config.read('./config/ai_alpha_config.ini')
alpha_config = config["AIAlpha"]

# These are the stocks to use
snp_500_stocks = utils.get_snp500()

# This is the price histories for those stocks
price_histories = phh.from_yahoo_finance(symbols=snp_500_stocks.index.to_list(),
                                         period=config_helper.get_number_of_years_of_price_histories(alpha_config),
                                         storage_path=config_helper.get_price_histories_path(alpha_config),
                                         reload=False)

# We will need a sector helper for the alphas
sector_helper = afh.get_sector_helper(snp_500_stocks, price_histories)

# These are the alpha factors
alpha_factors_df = afh.get_alpha_factors(price_histories, sector_helper,
                                         storage_path=config_helper.get_alpha_factors_path(alpha_config), reload=False)

# Train a model to generate the AI Alpha
ai_alpha_model = afh.get_ai_alpha_model(alpha_factors_df,
                                        price_histories,
                                        float(alpha_config['min_sharpe_ratio']),
                                        int(alpha_config['ForwardPredictionDays']),
                                        int(alpha_config['PredictionQuantiles']),
                                        int(alpha_config['RandomForestNTrees']),
                                        storage_path=config_helper.get_ai_model_path(alpha_config),
                                        reload=False)

# Create AI Alpha factors using the model and existing alpha factors
ai_alpha_name = alpha_config['AIAlphaName']
ai_alpha_df = afh.get_ai_alpha_factor(alpha_factors_df,
                                      ai_alpha_model,
                                      ai_alpha_name,
                                      storage_path=config_helper.get_ai_alpha_path(alpha_config),
                                      reload=False)

# Make some plots
# TODO: Checkout alphalense tear sheets examples
factor_returns_data, clean_factor_data, unix_time_factor_data = alpha_factors.evaluate_alpha(ai_alpha_df,
                                                                                             price_histories.Close)
alpha_factors.plot_factor_returns(factor_returns_data)
plt.show()
alpha_factors.plot_factor_rank_autocorrelation(clean_factor_data)
plt.show()
alpha_factors.plot_basis_points_per_day_quantile(unix_time_factor_data)
plt.show()

# Back testing phase

alpha_vectors = ai_alpha_df.copy().reset_index().pivot(index='Date', columns='Symbols', values=ai_alpha_name)
# if alpha_vectors needs to be read/stored:
#   pd.read_csv(storage_path, parse_dates=['Date']).set_index(['Date']).sort_index()

# TODO: Backtest
# TODO: Evaluate and push to prod for use
