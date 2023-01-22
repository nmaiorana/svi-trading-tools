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
ai_alpha_model = afh.get_ai_alpha_model(alpha_factors_df,
                                        price_histories,
                                        float(alpha_config['min_sharpe_ratio']),
                                        int(alpha_config['ForwardPredictionDays']),
                                        int(alpha_config['PredictionQuantiles']),
                                        int(alpha_config['RandomForestNTrees']),
                                        storage_path=config_helper.get_ai_model_path(alpha_config),
                                        reload=False)

ai_alpha_name = alpha_config['AIAlphaName']
ai_alpha = afh.get_ai_alpha_vector(alpha_factors_df,
                                   ai_alpha_model,
                                   ai_alpha_name,
                                   storage_path=config_helper.get_ai_alpha_path(alpha_config),
                                   reload=False)
factor_returns_data, clean_factor_data, unix_time_factor_data = alpha_factors.evaluate_alpha(ai_alpha,
                                                                                             price_histories.Close)
alpha_factors.plot_factor_returns(factor_returns_data)
alpha_factors.plot_factor_rank_autocorrelation(clean_factor_data)
alpha_factors.plot_basis_points_per_day_quantile(unix_time_factor_data)
cumulative_factor_returns = (1 + factor_returns_data).cumprod()
total_return = cumulative_factor_returns.iloc[-1].values[0]
logger.info(f'Total return on {ai_alpha.name} is {total_return}')

plt.show()

# TODO: Save Factors
# TODO: Create Configuration for factors_to_use for model training features
# TODO: Train model on factors_to_use
# TODO: Score model
# TODO: Backtest
# TODO: Evaluate and push to prod for use
