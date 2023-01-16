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


price_histories = phh.from_yahoo_finance_config(alpha_config, reload=True)
sector_helper = afh.get_sector_helper(alpha_config, price_histories.Close)
alpha_factors_file_name = alpha_config["DataDirectory"] + '/' + alpha_config["AlphaFactorsFileName"]

scored_factors = afh.generate_scored_factors(price_histories)
fixed_factors = afh.generate_fixed_factors(price_histories)

min_sharpe_ratio = float(alpha_config['min_sharpe_ratio'])
logger.info(f'FACTOR_EVAL|MIN_SHARPE_RATIO|{min_sharpe_ratio}')

all_factors_df = pd.concat(scored_factors + fixed_factors, axis=1)

# TODO: Add standard factors
# TODO: Save Factors
# TODO: Create Configuration for factors used for model training features
