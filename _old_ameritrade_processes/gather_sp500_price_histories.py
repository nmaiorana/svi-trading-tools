import logging.config
import configparser
from platform import python_version
import tools.price_histories_helper as phh

logging.config.fileConfig('./config/logging.ini')
logger = logging.getLogger('get_sp500_price_histories')
config = configparser.ConfigParser()
config.read('./config/ai_alpha_config.ini')
default_config = config["DEFAULT"]

logger.info(f'Getting stock histories for S&P 500')
logger.info(f'Python version: {python_version()}')
phh.from_yahoo_finance_config(default_config, reload=True)
logger.info(f'Gathered stock histories for S&P 500')

