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

import tools.price_histories_helper as phh
import tools.trading_factors_yahoo as alpha_factors
import tools.alpha_factors_helper as afh
import tools.backtesting_functions as btf
import tools.configuration_helper as config_helper
import tools.utils as utils
import warnings

from portfolio_optimizer import OptimalHoldings

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
backtesting_config = config["BackTest"]

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
# TODO: Backtest

alpha_vectors = btf.get_alpha_vectors(alpha_factors_df,
                                      config_helper.get_alpha_vectors_path(alpha_config),
                                      reload=False)
daily_betas = btf.generate_beta_factors(price_histories,
                                        config_helper.get_number_of_years_of_price_histories_int(alpha_config),
                                        config_helper.get_daily_betas_path(alpha_config),
                                        reload=False)


def get_lambda(average_dollar_volume):
    average_dollar_volume_cleaned = average_dollar_volume.replace(np.nan, 1.0e4)
    average_dollar_volume_cleaned = average_dollar_volume_cleaned.replace(0.0, 1.0e4)
    return 0.1 / average_dollar_volume_cleaned


def get_total_transaction_costs(h0, h_star, total_cost_lambda):
    return np.dot((h_star - h0) ** 2, total_cost_lambda)


risk_cap = float(backtesting_config['risk_cap'])
weights_max = float(backtesting_config['weights_max'])
weights_min = float(backtesting_config['weights_min'])
logger.info(f'OPTIMIZATION|risk_cap|{risk_cap}')
logger.info(f'OPTIMIZATION|weights_max|{weights_max}')
logger.info(f'OPTIMIZATION|weights_min|{weights_min}')

returns = alpha_factors.FactorReturns(price_histories).factor_data
adv = alpha_factors.AverageDollarVolume(price_histories, int(alpha_config['ForwardPredictionDays'])).factor_data
daily_return_n_days_delay = int(alpha_config['ForwardPredictionDays'])
delayed_returns = returns[-252:].shift(-daily_return_n_days_delay).dropna()
start_date = list(delayed_returns.index)[0]
end_date = list(delayed_returns.index)[-1]
logger.info(f'OPT|{start_date}|{end_date}')
tc_lambda = get_lambda(adv)

current_holdings = pd.Series(np.zeros(len(delayed_returns.columns)), index=delayed_returns.columns)

min_viable_port_return = float(backtesting_config['min_viable_port_return'])
opt_date_returns = {}
opt_date_tc = {}
for opt_date in tqdm(delayed_returns.index.to_list()[-252::daily_return_n_days_delay], desc='Dates',
                     unit=' Portfolio Optimization'):
    alpha_vector = pd.DataFrame(alpha_vectors.loc[opt_date])
    risk_model = daily_betas[opt_date.strftime('%m/%d/%Y')]
    est_return = delayed_returns.loc[opt_date]
    optimal_weights = OptimalHoldings(risk_cap=risk_cap, weights_max=weights_max,
                                      weights_min=weights_min).find(alpha_vector,
                                                                    risk_model.factor_betas_,
                                                                    risk_model.factor_cov_matrix_,
                                                                    risk_model.idiosyncratic_var_vector_)
    new_holdings = optimal_weights['optimalWeights']
    opt_date_returns[opt_date] = (new_holdings * est_return).sum()
    # trading costs
    opt_date_tc[opt_date] = get_total_transaction_costs(current_holdings, new_holdings, tc_lambda.loc[opt_date])
    current_holdings = new_holdings

port_return = round(np.sum(list(opt_date_returns.values())) * 100, 2)
logger.info(f'OPT_PORT_RETURN|{port_return}%')
pd.Series(opt_date_returns).cumsum().plot()
if port_return >= min_viable_port_return:
    logger.info(f'OPT|PROCEED|{port_return}%')
else:
    logger.warning(f'OPT|STOP|{port_return}')
    raise RuntimeError(f'Backtest indicates this strategy needs more work! ({port_return})') from None

# TODO: Evaluate and push to prod for use
