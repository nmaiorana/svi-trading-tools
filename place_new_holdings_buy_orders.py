import math
from platform import python_version
import configparser
import logging.config
import pandas as pd

import tools.configuration_helper as config_helper
import tools.ameritrade_functions as amc

"""
Use this script if you want to sell off all short-term holdings in your listed accounts.
Short-term holdings are defined by NOT being long_term_asset_types or long_term_stocks as configured for each account.
This is defined in a way that allows for short-term assets to be more dynamic.
"""

logging.config.fileConfig('./config/logging.ini')
main_logger_name = 'PlaceNewHoldingsBuyOrders'
logger = logging.getLogger(main_logger_name)
logger.info(f'Python version: {python_version()}')
logger.info(f'Pandas version: {pd.__version__}')

config = configparser.ConfigParser()
config.read('./config/config.ini')

default_config = config['DEFAULT']
logger.info(f'Getting TD Ameritrade...')
td_ameritrade = amc.AmeritradeRest()
accounts = config_helper.get_accounts(default_config)
for account in accounts:
    td_ameritrade.refresh_data()
    logger = logging.getLogger(f'{main_logger_name}.{account}')
    logger.info(f'Processing account {account}...')
    account_config = config[account]
    min_shares_to_buy = account_config.getint('MinSharesToBuy')
    adjusted_holdings_path = config_helper.get_data_directory(account_config).joinpath(account + '_adjusted_holdings.parquet')
    stocks_to_buy = pd.read_parquet(adjusted_holdings_path)
    masked_account_number = config_helper.get_masked_account_number(account_config)
    total_portfolio_value = round(td_ameritrade.get_account_value(masked_account_number), 2)
    logger.info(f'MASKED_ACCOUNT_NUMBER|{masked_account_number}|VALUE|{total_portfolio_value}')
    current_cash_balance = td_ameritrade.parse_accounts().loc[masked_account_number].currentBalances_cashBalance
    cash_equivalents_balance = td_ameritrade.get_account_portfolio_data(masked_account_number,
                                                                        'CASH_EQUIVALENT').marketValue.sum()
    total_amount_available = current_cash_balance + cash_equivalents_balance
    total_amount_available = min(total_amount_available, config_helper.get_max_investment_amount(account_config))
    investment_base = 1000
    investment_amount = math.floor(total_amount_available / investment_base) * investment_base
    logger.info(f'ACCOUNT|{masked_account_number}|CASH_VALUE|{current_cash_balance}|' +
                f'CASH_EQUIV|{cash_equivalents_balance}|' +
                f'AVAILABLE|{total_amount_available}|' +
                f'USING|{investment_amount}')

    if investment_amount <= 0:
        continue
    trade_configurations_df = stocks_to_buy
    trade_configurations_df['Amount'] = (trade_configurations_df.optimalWeights * investment_amount).round(0)
    quotes_df = td_ameritrade.get_quotes(list(trade_configurations_df.index.to_list()))[
        ['assetType', 'bidPrice', 'askPrice', 'regularMarketLastPrice']]
    trade_configurations_df = pd.concat([trade_configurations_df, quotes_df], axis=1)
    trade_configurations_df['Quantity'] = (
                trade_configurations_df.Amount / trade_configurations_df.regularMarketLastPrice).round(0)

    trade_configurations_df.fillna(0)
    for symbol, row in trade_configurations_df.iterrows():
        if pd.isna(row.Quantity) or row.Quantity == 0:
            continue
        instruction = 'BUY'
        quantity = int(abs(row.Quantity))
        if quantity <= min_shares_to_buy:
            logger.warning(f'Trade for {symbol} is less than {min_shares_to_buy}, skipping...')
            continue
        ask_price = round(row.regularMarketLastPrice, 2)

        order = amc.create_limit_order(masked_account_number,
                                       symbol, 'EQUITY', quantity, instruction, 'NORMAL', 'DAY', ask_price)
        logger.info(f'ORDER|{order}')
        td_ameritrade.place_order(order, saved=False)

