import logging.config
import configparser
import math
from platform import python_version
import pandas as pd
import matplotlib.pyplot as plt

import tools.price_histories_helper as phh
import tools.alpha_factors_helper as afh
import tools.backtesting_functions as btf
import tools.configuration_helper as config_helper
import tools.utils as utils
import tools.ameritrade_functions as amc
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
    ai_alpha_model = afh.load_ai_alpha_model(storage_path=config_helper.get_ai_model_final_path(strategy_config))

    # Create AI Alpha factors using the model and existing alpha factors
    ai_alpha_name = strategy_config['AIAlphaName']
    ai_alpha_df = afh.get_ai_alpha_factor(alpha_factors_df,
                                          ai_alpha_model,
                                          ai_alpha_name,
                                          storage_path=config_helper.get_ai_alpha_final_path(strategy_config),
                                          reload=ai_alpha_factor_reload)
    alpha_vectors = btf.get_alpha_vectors(alpha_factors_df,
                                          config_helper.get_alpha_vectors_final_path(strategy_config),
                                          reload=alpha_vectors_reload)
    daily_betas = btf.generate_beta_factors(price_histories,
                                            config_helper.get_number_of_years_of_price_histories_int(portfolio_config),
                                            config_helper.get_number_of_risk_exposures(strategy_config),
                                            config_helper.get_daily_betas_final_path(strategy_config),
                                            reload=daily_betas_reload)
td_ameritrade = amc.AmeritradeRest()
for account in accounts:
    # portfolio adjustments
    account_config = config[account]
    min_shares_to_buy = account_config.getint('MinSharesToBuy')
    implemented_strategy = config_helper.get_implemented_strategy(account_config)
    logger = logging.getLogger(f'GenerateAndSelectAlphaFactors.{implemented_strategy}.adjustment')
    logger.info(f'Adjusting portfolio: {account}')
    strategy_config_parser = configparser.ConfigParser()
    strategy_config_parser.read('./data/' + implemented_strategy + '/config.ini')
    strategy_config = strategy_config_parser['DEFAULT']
    final_strategy_path = config_helper.get_final_strategy_path(strategy_config)
    alpha_vectors = btf.load_alpha_vectors(config_helper.get_alpha_vectors_final_path(strategy_config))
    daily_betas = btf.load_beta_factors(config_helper.get_daily_betas_final_path(strategy_config))
    min_viable_return = strategy_config.getfloat('min_viable_port_return')
    forward_prediction_days = strategy_config.getint('ForwardPredictionDays')
    optimal_holdings_df = btf.predict_optimal_holdings(alpha_vectors,
                                                       daily_betas,
                                                       list(daily_betas.keys())[-1:],
                                                       risk_cap=strategy_config.getfloat('risk_cap'),
                                                       weights_max=strategy_config.getfloat('weights_max'),
                                                       weights_min=strategy_config.getfloat('weights_min'))

    optimal_holdings = optimal_holdings_df.iloc[-1].round(2)
    optimal_holdings.name = 'optimalWeights'
    optimal_holdings = optimal_holdings[optimal_holdings > 0.05]
    for index, value in optimal_holdings.items():
        logger.info(f'STOCK|{index:20}|HOLDING|{value:2f}')
    for index, row in td_ameritrade.get_quotes(list(optimal_holdings.index.to_list())).iterrows():
        logger.info(f'QUOTES|{row.symbol}|{row.assetMainType}|CUSIP|{row.cusip}|' +
                    f'BID/ASK|{row.bidPrice}/{row.askPrice}|CHANGE|{row.regularMarketNetChange}' +
                    f'|{row.description}')
    td_ameritrade.refresh_data()
    masked_account_number = config_helper.get_masked_account_number(account_config)
    current_cash_balance = td_ameritrade.parse_accounts().loc[masked_account_number].currentBalances_cashBalance
    cash_equivalents_balance = td_ameritrade.get_account_portfolio_data(masked_account_number,
                                                                        'CASH_EQUIVALENT').marketValue.sum()
    total_amount_available = current_cash_balance + cash_equivalents_balance
    investment_base = 1000
    investment_amount = math.floor(total_amount_available / investment_base) * investment_base
    logger.info(f'ACCOUNT|{masked_account_number}|CASH_VALUE|{current_cash_balance}|' +
                f'CASH_EQUIV|{cash_equivalents_balance}|' +
                f'AVAILABLE|{total_amount_available}|' +
                f'USING|{investment_amount}')
    # TODO: Make stock trades
    trade_configurations_df = optimal_holdings.to_frame()
    trade_configurations_df['Amount'] = (trade_configurations_df.optimalWeights * investment_amount).round(0)
    quotes_df = td_ameritrade.get_quotes(list(trade_configurations_df.index.to_list()))[
        ['assetType', 'bidPrice', 'askPrice', 'regularMarketLastPrice']]
    trade_configurations_df = pd.concat([trade_configurations_df, quotes_df], axis=1)
    trade_configurations_df['Quantity'] = (
                trade_configurations_df.Amount / trade_configurations_df.regularMarketLastPrice).round(0)
    for symbol, row in trade_configurations_df.iterrows():

        instruction = 'BUY'
        quantity = int(abs(row.Quantity))
        if quantity <= min_shares_to_buy:
            logger.warning(f'Trade for {symbol} is less than {min_shares_to_buy}, skipping...')
            continue
        ask_price = round(row.regularMarketLastPrice, 2)

        order = amc.create_limit_order(masked_account_number,
                                       symbol, 'EQUITY', quantity, instruction, 'NORMAL', 'DAY', ask_price)
        logger.info(f'ORDER|{order}')
        td_ameritrade.place_order(order, saved=True)
