import logging
import logging.config


import configparser
from platform import python_version
from pathlib import Path

# Project imports
import tools.trading_factors_yahoo as alpha_factors
import tools.utils as utils
import tools.nonoverlapping_estimator as ai_estimator

import time
from datetime import datetime
import os
import pandas as pd
import numpy as np
import math
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger('GenerateAIAlphaFactor')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20, 8)

logger.info(f'Python version: {python_version()}')
logger.info(f'Pandas version: {pd.__version__}')

config = configparser.ConfigParser()
config.read('./config/ai_alpha_config.ini')
default_config = config['AIAlpha']

price_histories_file_name = default_config['DataDirectory'] + '/' + default_config['PriceHistoriesFileName']
logger.info(f'PRICE_HISTORIES_FILE|{price_histories_file_name}...')
price_histories = pd.read_csv(price_histories_file_name, header=[0, 1], index_col=[0], parse_dates=True, low_memory=False)
logger.info(f'PRICE_HISTORIES|{price_histories.index.min()}|{price_histories.index.max()}')
logger.info(f'Using {default_config["NumberOfYearsForAlpha"]} years of price history data to generate alpha factors.')
latest_date = price_histories.index.max() 
earliest_date = latest_date - pd.DateOffset(years=int(default_config["NumberOfYearsForAlpha"]))
price_histories = price_histories[(price_histories.index >= earliest_date) & (price_histories.index <= latest_date)]
logger.info(f'PRICE_HISTORIES_ALPHA|{price_histories.index.min()}|{price_histories.index.max()}')
close = price_histories.Close
logger.info(f'STOCK_TICKERS|{len(close.columns)}')
alpha_factors_file_name = default_config['DataDirectory'] + '/' + default_config['AlphaFactorsFileName']

logger.info(f'ALPHA_FACTORS_FILE|{alpha_factors_file_name}')
all_factors = pd.read_csv(alpha_factors_file_name, parse_dates=['Date']).set_index(['Date', 'Symbols']).sort_index()
logger.info('Alpha factors read.')

for alpha_factor in all_factors.columns:
    logger.info(f'ALPHA_FACTOR|{alpha_factor}')


# # Stage 3: Generate AI Alpha Factor
# 
# - Compute target values (y)
#     - Quantize with 2 bins
# - Train model for Feature importance
# - Feature reduction
# - Train model for AI Alpha Vector
# - Compute AI Alpha Vectors for 1 year
# - Save AI Alpha Vectors

# ## Compute the target values (y) and Shift back to create a 5 day forward prediciton
# 
# This is something you want to experiment with. If you are planning on holding on to assets for long periods of
# time, perhaps a 20, 40 or 60 forward prediction will work better.

forward_prediction_days = int(default_config['ForwardPredictionDays'])
prod_target_quantiles = int(default_config['PredictionQuantiles'])
prod_target_source = f'{forward_prediction_days}Day{prod_target_quantiles}Quant'

logger.info(f'Setting {forward_prediction_days} days-{prod_target_quantiles} quantiles to target {prod_target_source}')
all_assets = all_factors.index.levels[1].values.tolist()
logger.info(f'Factors from date: {all_factors.index.levels[0].min()} to date: {all_factors.index.levels[0].max()}')
features = all_factors.columns.tolist()

training_factors = pd.concat(
[
    all_factors,
    alpha_factors.FactorReturnQuantiles(price_histories, prod_target_quantiles, forward_prediction_days).for_al(prod_target_source),
], axis=1).dropna()
training_factors.sort_index(inplace=True)

training_factors['target'] = training_factors.groupby(level=1)[prod_target_source].shift(-forward_prediction_days)

logger.info(f'Creating training and label data...')
temp = training_factors.dropna().copy()
X = temp[features]
y = temp['target']
for feature in features:
    logger.info(f'TRAINING_FEATURE|{feature}')
logger.info(f'TRAINING_DATASET|{len(X)}|LABEL_DATASET|{len(y)}')

n_days = 10
n_stocks = len(set(all_factors.index.get_level_values(level=1).values))
clf_random_state = 42

clf_parameters = {
    'criterion': 'entropy',
    'min_samples_leaf': n_days * n_stocks,
    'oob_score': True,
    'n_jobs': -1,
    'random_state': clf_random_state}

n_trees = int(default_config['RandomForestNTrees'])

logger.info(f'Creating RandomForestClassifier with {n_trees} trees...')
for key, value in clf_parameters.items():
    logger.info(f'Parameter: {key} set to {value}')
clf = RandomForestClassifier(n_trees, **clf_parameters)

logger.info(f'Creating Non-Overlapping Voter with {forward_prediction_days - 1} non-overlapping windows...')
clf_nov = ai_estimator.NoOverlapVoter(clf, n_skip_samples=forward_prediction_days - 1)

logger.info(f'Training classifier...')
clf_nov.fit(X, y)

logger.info(f'CLASSIFIER|TRAIN|{clf_nov.score(X, y.values)}|OOB|{clf_nov.oob_score_}')
ai_alpha_name = default_config['AIAlphaName']
logger.info(f'AIAlpha|GET_SCORE|{ai_alpha_name}')
factors_with_alpha = alpha_factors.add_alpha_score(training_factors[features].copy(), clf_nov, ai_alpha_name)
logger.info(f'Factors with AIAlpha from date: {factors_with_alpha.index.levels[0].min()} to date: {factors_with_alpha.index.levels[0].max()}')
for alpha_factor in factors_with_alpha.columns:
    logger.info(f'ALPHA_FACTOR|{alpha_factor}')

factors_to_compare = features
_ = alpha_factors.evaluate_alpha(factors_with_alpha[[ai_alpha_name]], close)
ai_alpha = factors_with_alpha[ai_alpha_name].copy()
alpha_vectors = ai_alpha.reset_index().pivot(index='Date', columns='Symbols', values=ai_alpha_name)
ai_alpha_factors_file_name = default_config['DataDirectory'] + '/' + default_config['AIAlphaFileName']
logger.info(f'AI_ALPHA_FACTORS_FILE|{ai_alpha_factors_file_name}')
alpha_vectors.reset_index().to_csv(ai_alpha_factors_file_name, index=False)

pre_backtest_model_file_name = default_config['DataDirectory'] + '/' + default_config['PreBacktestModelFileName']

with open(pre_backtest_model_file_name, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(clf_nov, f, pickle.HIGHEST_PROTOCOL)

