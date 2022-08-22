import gather_sp500_price_histories
import logging.config
import subprocess

logging.config.fileConfig('./config/logging.ini')

subprocess.call("gather_sp500_price_histories.py", shell=False)
subprocess.call("generate_alpha_beta_factors.py", shell=False)
