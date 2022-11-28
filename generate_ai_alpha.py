import configparser
import logging
import logging.config
import pandas as pd
import pickle
import tools.trading_factors_yahoo as alpha_factors
import generate_alpha_beta_factors


config = configparser.ConfigParser()
config.read('./config/config.ini')
default_config = config['AIAlpha']

generate_alpha_beta_factors.generate_alpha_beta_factors()
logging.config.fileConfig('./config/logging.ini')
logger = logging.getLogger('GenerateAIAlphaFactor')
alpha_factors_file_name = default_config['DataDirectory'] + '/' + default_config['AlphaFactorsFileName']

logger.info(f'ALPHA_FACTORS_FILE|{alpha_factors_file_name}')
all_factors = pd.read_csv(alpha_factors_file_name, parse_dates=['Date']).set_index(['Date', 'Symbols']).sort_index()
logger.info('Alpha factors read.')

for alpha_factor in all_factors.columns:
    logger.info(f'ALPHA_FACTOR|{alpha_factor}')


pre_backtest_model_file_name = default_config['DataDirectory'] + '/' + default_config['ModelFileName']
logger.info(f'Reading AI Alpha Model: {pre_backtest_model_file_name}')
clf_nov = pickle.load(open(pre_backtest_model_file_name, 'rb'))

factors_with_alpha = alpha_factors.add_alpha_score(all_factors, clf_nov, default_config['AIAlphaName'])
ai_alpha = factors_with_alpha[default_config['AIAlphaName']].copy()
alpha_vectors = ai_alpha.reset_index().pivot(index='Date', columns='Symbols', values=default_config['AIAlphaName'])
ai_alpha_factors_file_name = default_config['DataDirectory'] + '/' + default_config['AIAlphaFileName']
logger.info(f'AI_ALPHA_FACTORS_FILE|{ai_alpha_factors_file_name}')
alpha_vectors.reset_index().to_csv(ai_alpha_factors_file_name, index=False)






