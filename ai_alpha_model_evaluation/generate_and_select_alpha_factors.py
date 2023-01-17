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
alpha_config = config["Alpha"]

price_histories = phh.from_yahoo_finance_config(alpha_config, reload=False)
close = price_histories.Close
sector_helper = afh.get_sector_helper(alpha_config, price_histories.Close)

alpha_factors_file_name = config_helper.get_alpha_factors_path(alpha_config)

alpha_factors_df = afh.generate_factors(price_histories, sector_helper)

min_sharpe_ratio = float(alpha_config['min_sharpe_ratio'])
logger.info(f'FACTOR_EVAL|MIN_SHARPE_RATIO|{min_sharpe_ratio}')
factors_to_use = afh.identify_factors_to_use(alpha_factors_df, close, min_sharpe_ratio)
for factor_name in factors_to_use:
    logger.info(f'SELECTED_FACTOR|{factor_name}')

ai_alpha_model = afh.train_ai_alpha_model(alpha_factors_df[factors_to_use], price_histories)
# TODO: Save Factors
# TODO: Create Configuration for factors_to_use for model training features
# TODO: Train model on factors_to_use
# TODO: Score model
# TODO: Backtest
# TODO: Evaluate and push to prod for use
