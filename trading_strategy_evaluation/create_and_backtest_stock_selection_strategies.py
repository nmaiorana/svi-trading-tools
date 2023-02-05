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

# See which artifacts need to be reproduced based on the upstream artifacts
price_histories_reload = False
alpha_factors_reload = False
ai_alpha_model_reload = False
ai_alpha_factor_reload = False
alpha_vectors_reload = False
daily_betas_reload = False
backtest_factors_reload = False

if not config_helper.get_price_histories_path(alpha_config).exists():
    alpha_factors_reload = True
    ai_alpha_model_reload = True
    ai_alpha_factor_reload = True
    alpha_vectors_reload = True
    daily_betas_reload = True
    backtest_factors_reload = True

if not config_helper.get_alpha_factors_path(alpha_config).exists():
    ai_alpha_model_reload = True
    ai_alpha_factor_reload = True
    alpha_vectors_reload = True
    daily_betas_reload = True
    backtest_factors_reload = True

# This is the price histories for those stocks
price_histories = phh.from_yahoo_finance(symbols=snp_500_stocks.index.to_list(),
                                         period=config_helper.get_number_of_years_of_price_histories(alpha_config),
                                         storage_path=config_helper.get_price_histories_path(alpha_config),
                                         reload=price_histories_reload)

# We will need a sector helper for the alphas
sector_helper = afh.get_sector_helper(snp_500_stocks, price_histories)

# These are the alpha factors
alpha_factors_df = afh.get_alpha_factors(price_histories, sector_helper,
                                         storage_path=config_helper.get_alpha_factors_path(alpha_config),
                                         reload=alpha_factors_reload)
evaluation_strategies = alpha_config['Strategies'].split()
for strategy in evaluation_strategies:
    logger = logging.getLogger(f'GenerateAndSelectAlphaFactors.{strategy}')
    strategy_config = config[strategy]
    eval_strategy_path = config_helper.get_strategy_path(strategy_config)
    final_strategy_path = config_helper.get_final_strategy_path(strategy_config)
    logger.info('**********************************************************************************************')
    logger.info(f'Checking existing strategy completion: {final_strategy_path}')
    if final_strategy_path.exists():
        logger.info(f'Final strategy for {strategy} already rendered {final_strategy_path}')
        continue

    logger.info(f'Constructing and Evaluating strategy {strategy}')
    # Check for reloading of downstream artifacts
    if not config_helper.get_ai_model_path(strategy_config).exists():
        ai_alpha_factor_reload = True
        alpha_vectors_reload = True
        daily_betas_reload = True
        backtest_factors_reload = True

    # Train a model to generate the AI Alpha
    ai_alpha_model = afh.get_ai_alpha_model(alpha_factors_df,
                                            price_histories,
                                            float(strategy_config['min_sharpe_ratio']),
                                            int(strategy_config['ForwardPredictionDays']),
                                            int(strategy_config['PredictionQuantiles']),
                                            int(strategy_config['RandomForestNTrees']),
                                            storage_path=config_helper.get_ai_model_path(strategy_config),
                                            reload=ai_alpha_model_reload)

    # Create AI Alpha factors using the model and existing alpha factors
    ai_alpha_name = strategy_config['AIAlphaName']
    ai_alpha_df = afh.get_ai_alpha_factor(alpha_factors_df,
                                          ai_alpha_model,
                                          ai_alpha_name,
                                          storage_path=config_helper.get_ai_alpha_path(strategy_config),
                                          reload=ai_alpha_factor_reload)

    # Make some plots
    # TODO: Checkout alphalense tear sheets examples
    # TODO: Store the evaluation artifacts so they can be looked at in a better setting, like Jupyter Labs
    factor_returns_data, clean_factor_data, unix_time_factor_data = alpha_factors.evaluate_alpha(ai_alpha_df,
                                                                                                 price_histories.Close)
    factors_sharpe_ratio = alpha_factors.compute_sharpe_ratio(factor_returns_data)

    # Show Results
    logger.info(f'SHARPE_RATIO|{ai_alpha_name}|{factors_sharpe_ratio.loc[ai_alpha_name].round(2)}')
    alpha_factors.plot_factor_returns(factor_returns_data)
    plt.show()
    alpha_factors.plot_basis_points_per_day_quantile(clean_factor_data)
    plt.show()

    # Back testing phase
    alpha_vectors = btf.get_alpha_vectors(alpha_factors_df,
                                          config_helper.get_alpha_vectors_path(strategy_config),
                                          reload=alpha_vectors_reload)
    daily_betas = btf.generate_beta_factors(price_histories,
                                            config_helper.get_number_of_years_of_price_histories_int(strategy_config),
                                            config_helper.get_number_of_risk_exposures(strategy_config),
                                            config_helper.get_daily_betas_path(strategy_config),
                                            reload=daily_betas_reload)
    min_viable_return = float(strategy_config['min_viable_port_return'])
    net_returns, optimal_holdings_df = btf.backtest_factors(price_histories,
                                                            alpha_vectors,
                                                            daily_betas,
                                                            int(strategy_config['ForwardPredictionDays']),
                                                            backtest_days=int(126),
                                                            risk_cap=float(strategy_config['risk_cap']),
                                                            weights_max=float(strategy_config['weights_max']),
                                                            weights_min=float(strategy_config['weights_min']))
    optimal_holdings = optimal_holdings_df.iloc[-1].round(2)
    optimal_holdings = optimal_holdings[optimal_holdings > 0.05]
    for index, value in optimal_holdings.items():
        logger.info(f'STOCK|{index:20}|HOLDING|{value:2f}')

    net_returns.cumsum().plot(title=strategy + ' Backtest Returns')
    plt.show()
    port_return = round(net_returns.sum() * 100, 2)
    logger.info(f'OPT_PORT_RETURN|{port_return}%')
    if port_return >= min_viable_return:
        logger.info(f'OPT|PROCEED|{port_return}% >= {min_viable_return}%')
        strategy_config_path = config_helper.get_strategy_config_path(strategy_config)
        logger.info(f'Saving strategy configuration to {strategy_config_path}')
        strategy_config_parser = configparser.ConfigParser(strategy_config)
        with open(strategy_config_path, 'w') as configfile:
            strategy_config_parser.write(configfile)
        logger.info(f'Rendering final strategy {final_strategy_path}')
        eval_strategy_path.replace(final_strategy_path)
    else:
        logger.warning(f'OPT|STOP|{port_return}% < {min_viable_return}%')
        raise RuntimeError(f'Backtest indicates this strategy needs more work! ({port_return}%)') from None
    # TODO: Store port_return data.
    # TODO: Store a config file in the strategy to match all the parameters needed in prod
