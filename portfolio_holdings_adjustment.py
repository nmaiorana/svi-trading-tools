import logging.config
import configparser
from platform import python_version
import pandas as pd
import matplotlib.pyplot as plt

import tools.price_histories_helper as phh
import tools.trading_factors_yahoo as alpha_factors
import tools.alpha_factors_helper as afh
import tools.backtesting_functions as btf
import tools.configuration_helper as config_helper
import tools.utils as utils
import warnings


warnings.filterwarnings('ignore')

logging.config.fileConfig('./config/logging.ini')

logger = logging.getLogger('PortfolioHoldingsAdjustment')

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20, 8)

logger.info(f'Python version: {python_version()}')
logger.info(f'Pandas version: {pd.__version__}')

config = configparser.ConfigParser()
config.read('./config/config.ini')
portfolio_config = config["DEFAULT"]

# These are the stocks to use
snp_500_stocks = utils.get_snp500()

# See which artifacts need to be reproduced based on the upstream artifacts
price_histories_reload = False
alpha_factors_reload = False
ai_alpha_model_reload = False
ai_alpha_factor_reload = False
alpha_vectors_reload = False
daily_betas_reload = False
backtest_factors_reload = False


if not config_helper.get_price_histories_path(portfolio_config).exists():
    alpha_factors_reload = True
    ai_alpha_model_reload = True
    ai_alpha_factor_reload = True
    alpha_vectors_reload = True
    daily_betas_reload = True
    backtest_factors_reload = True

if not config_helper.get_alpha_factors_path(portfolio_config).exists():
    ai_alpha_model_reload = False
    ai_alpha_factor_reload = True
    alpha_vectors_reload = True
    daily_betas_reload = True
    backtest_factors_reload = True


# This is the price histories for those stocks
price_histories = phh.from_yahoo_finance(symbols=snp_500_stocks.index.to_list(),
                                         period=config_helper.get_number_of_years_of_price_histories(portfolio_config),
                                         storage_path=config_helper.get_price_histories_path(portfolio_config),
                                         reload=price_histories_reload)

# We will need a sector helper for the alphas
sector_helper = afh.get_sector_helper(snp_500_stocks, price_histories)

# These are the alpha factors
alpha_factors_df = afh.get_alpha_factors(price_histories, sector_helper,
                                         storage_path=config_helper.get_alpha_factors_path(portfolio_config),
                                         reload=alpha_factors_reload)


# TODO: Go through all the portfolio configurations and pull out the strategies used
# TODO: With a list of strategies to use, process the data to get alphas, ai_alpha, alpha vectors, daily betas
# TODO: With list of portfolios, generate the optimal holdings for each one using the selected strategy
accounts = config_helper.get_accounts(portfolio_config)
implemented_strategies = set()
for account in accounts:
    account_config = config[account]
    implemented_strategies.add(config_helper.get_implemented_strategy(account_config))

for strategy in implemented_strategies:
    logger = logging.getLogger(f'GenerateAndSelectAlphaFactors.{strategy}')
    strategy_config = configparser.ConfigParser()
    strategy_config.read('./data/' + strategy + '/config.ini')
    strategy_config = strategy_config['DEFAULT']
    final_strategy_path = config_helper.get_final_strategy_path(strategy_config)
    logger.info('**********************************************************************************************')
    logger.info(f'Updating strategy data: {final_strategy_path}')

    # Train a model to generate the AI Alpha
    ai_alpha_model = afh.load_ai_alpha_model(storage_path=config_helper.get_ai_model_path(strategy_config))

    # Create AI Alpha factors using the model and existing alpha factors
    ai_alpha_name = strategy_config['AIAlphaName']
    ai_alpha_df = afh.get_ai_alpha_factor(alpha_factors_df,
                                          ai_alpha_model,
                                          ai_alpha_name,
                                          storage_path=config_helper.get_ai_alpha_path(strategy_config),
                                          reload=ai_alpha_factor_reload)
    alpha_vectors = btf.get_alpha_vectors(alpha_factors_df,
                                          config_helper.get_alpha_vectors_path(strategy_config),
                                          reload=alpha_vectors_reload)
    daily_betas = btf.generate_beta_factors(price_histories,
                                            config_helper.get_number_of_years_of_price_histories_int(strategy_config),
                                            config_helper.get_number_of_risk_exposures(strategy_config),
                                            config_helper.get_daily_betas_path(strategy_config),
                                            reload=daily_betas_reload)

for account in accounts:
    # portfolio adjustments
    account_config = config[account]
    implemented_strategy = config_helper.get_implemented_strategy(account_config)
    strategy_config = configparser.ConfigParser()
    strategy_config.read('./data/' + implemented_strategy + '/config.ini')
    strategy_config = config['DEFAULT']
    final_strategy_path = config_helper.get_final_strategy_path(strategy_config)
    alpha_vectors = btf.load_alpha_vectors(config_helper.get_alpha_vectors_path(strategy_config))
    daily_betas = btf.load_beta_factors(config_helper.get_daily_betas_path(strategy_config))
    min_viable_return = float(strategy_config['min_viable_port_return'])
    net_returns, optimal_holdings = btf.backtest_factors(price_histories,
                                                         alpha_vectors,
                                                         daily_betas,
                                                         int(strategy_config['ForwardPredictionDays']),
                                                         backtest_days=int(1),
                                                         risk_cap=float(strategy_config['risk_cap']),
                                                         weights_max=float(strategy_config['weights_max']),
                                                         weights_min=float(strategy_config['weights_min']),
                                                         data_path=final_strategy_path,
                                                         reload=backtest_factors_reload)
    optimal_holdings = optimal_holdings[(100 * optimal_holdings['optimalWeights']).round() > 5.0]
    for index, row in optimal_holdings.iterrows():
        logger.info(f'STOCK|{index:20}|HOLDING|{row.optimalWeights:2f}')

    # TODO: Make stock trades
