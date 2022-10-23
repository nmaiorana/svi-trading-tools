import logging.config
from platform import python_version
import runpy

logging.config.fileConfig('./config/logging.ini')
logger = logging.getLogger('BuildAIAlphaModel')
logger.info(f'Python version: {python_version()}')

runpy.run_path('./gather_sp500_price_histories.py')

runpy.run_path('./generate_alpha_beta_factors.py')

