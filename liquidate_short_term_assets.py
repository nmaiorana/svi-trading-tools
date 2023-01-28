from IPython.core.display_functions import display
from platform import python_version
import configparser
import logging.config
import pandas as pd

import tools.configuration_helper as confi_helper
import tools.ameritrade_functions as amc

logging.config.fileConfig('./config/logging.ini')
logger = logging.getLogger('LiquidateShortTermAssets')
logger.info(f'Python version: {python_version()}')
logger.info(f'Pandas version: {pd.__version__}')

config = configparser.ConfigParser()
config.read('./config/config.ini')
default_config = config['DEFAULT']

accounts = confi_helper.get_accounts(default_config)
for account in accounts:
    logger.info(f'Processing account {account}')


'''td_ameritrade = amc.AmeritradeRest()
display(td_ameritrade.get_quotes(list(long_weights.index.to_list())))'''