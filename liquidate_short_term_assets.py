import math
from datetime import datetime
from platform import python_version
import configparser
import logging.config
import pandas as pd

import tools.configuration_helper as config_helper
import tools.ameritrade_functions as amc

logging.config.fileConfig('./config/logging.ini')
logger = logging.getLogger('LiquidateShortTermAssets')
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
    logger = logging.getLogger(f'{logger.name}.{account}')
    logger.info(f'Processing account {account}...')
    account_config = config[account]
    masked_account_number = config_helper.get_masked_account_number(account_config)
    total_portfolio_value = round(td_ameritrade.get_account_value(masked_account_number), 2)
    logger.info(f'MASKED_ACCOUNT_NUMBER|{masked_account_number}|VALUE|{total_portfolio_value}')

    for account_num, row in td_ameritrade.get_holdings(masked_account_number).iterrows():
        logger.info(f'{account_num}|MARKET_VALUE|{row.marketValue}|QUANTITY|{row.longQuantity}')

    long_term_stocks = config_helper.get_long_term_stocks(account_config)
    logger.info(f'LONG_TERM_STOCKS|{long_term_stocks}')

    long_term_asset_types = config_helper.get_long_term_asset_types(account_config)
    logger.info(f'LONG_TERM_ASSET_TYPES|{long_term_asset_types}')

    portfolio_stock_symbols = td_ameritrade.get_investment_symbols(masked_account_number, 'EQUITY')
    portfolio_fundamentals = td_ameritrade.get_fundamental(portfolio_stock_symbols)
    long_term_by_type = [fundamental[0] for fundamental in portfolio_fundamentals[['symbol', 'assetType']].values
                         if fundamental[1] in long_term_asset_types]
    stocks_to_trade = [symbol for symbol in portfolio_stock_symbols if symbol not in (long_term_stocks + long_term_by_type)]
    logger.info(f'STOCKS_TO_TRADE|{stocks_to_trade}')
    quotes_for_stocks = td_ameritrade.get_quotes(stocks_to_trade)
    for symbol, row in quotes_for_stocks.iterrows():
        logger.info(f'QUOTE|{symbol}|BID|{row.bidPrice}|ASK|{row.askPrice}|LAST|{row.regularMarketLastPrice}')

    current_holdings = td_ameritrade.get_holdings(masked_account_number, symbols=stocks_to_trade)
    current_holdings = current_holdings.droplevel('account')

    for symbol, row in current_holdings.iterrows():

        instruction = 'SELL'
        quantity = int(abs(row.longQuantity))
        ask_price = round(quotes_for_stocks.loc[symbol].regularMarketLastPrice, 2)

        order = amc.create_limit_order(masked_account_number, symbol, 'EQUITY', quantity, instruction, 'NORMAL',
                                       'DAY', ask_price)
        logger.info(f'ORDER|{order}')
        # td_ameritrade.place_order(order)
