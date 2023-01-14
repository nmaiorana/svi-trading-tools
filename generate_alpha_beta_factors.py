# Generate Alpha Factors
import logging
import logging.config
import configparser
from platform import python_version

import tools.trading_factors_yahoo as alpha_factors
import tools.utils as utils

import pickle
import pandas as pd
import pandas_datareader as pdr
from tqdm import tqdm


def generate_alpha_beta_factors():
    logging.config.fileConfig('./config/logging.ini')
    logger = logging.getLogger('GenerateAlphaAndBeta')
    logger.info(f'Python version: {python_version()}')
    logger.info(f'Pandas version: {pd.__version__}')
    logger.info(f'Pandas Data Reader version: {pdr.__version__}')
    config = configparser.ConfigParser()
    config.read('./config/config.ini')
    default_config = config["DEFAULT"]
    alpha_config = config["Alpha"]
    price_histories_file_name = alpha_config["DataDirectory"] + '/' + alpha_config["PriceHistoriesFileName"]
    logger.info(f'PRICE_HISTORIES_FILE|{price_histories_file_name}...')
    price_histories = pd.read_csv(price_histories_file_name, header=[0, 1], index_col=[0], parse_dates=True,
                                  low_memory=False)
    logger.info(f'PRICE_HISTORIES|{price_histories.index.min()}|{price_histories.index.max()}')
    logger.info(f'Using {alpha_config["NumberOfYearsForAlpha"]} years of price history data to generate alpha factors.')
    latest_date = price_histories.index.max()
    earliest_date = latest_date - pd.DateOffset(years=int(alpha_config["NumberOfYearsForAlpha"]))
    price_histories = price_histories[(price_histories.index >= earliest_date) & (price_histories.index <= latest_date)]
    logger.info(f'PRICE_HISTORIES_ALPHA|{price_histories.index.min()}|{price_histories.index.max()}')
    close = price_histories.Close
    logger.info(f'STOCK_TICKERS|{len(close.columns)}')
    alpha_factors_file_name = alpha_config["DataDirectory"] + '/' + alpha_config["AlphaFactorsFileName"]
    beta_factors_file_name = alpha_config["DataDirectory"] + '/' + alpha_config["BetaFactorsFileName"]
    logger.info('Gathering stock ticker sector data...')
    snp_500_stocks = utils.get_snp500()
    sector_helper = alpha_factors.get_sector_helper(snp_500_stocks, 'GICS Sector', close.columns)
    logger.info(f'Stock sector information gathered.')
    alpha_factors_list = [
        alpha_factors.TrailingOvernightReturns(price_histories, 10).rank().zscore().smoothed(
            10).rank().zscore().for_al(),
        alpha_factors.MarketDispersion(price_histories, 120).for_al(),
        alpha_factors.MarketVolatility(price_histories, 120).for_al(),
    ]

    for factor_used in alpha_factors_list:
        logger.info(f'USED_FACTOR|{factor_used.name}')

    logger.info(f'Combining {len(alpha_factors_list)} alphas into one dataframe...')
    all_factors = pd.concat(alpha_factors_list, axis=1)
    all_factors.sort_index(inplace=True)
    all_factors = all_factors.dropna()

    if len(all_factors) == 0:
        logger.error(f'ALPHA_FACTORS_EMPTY|{len(all_factors)}')
        raise RuntimeError(f'Alpha Factors contains no data({len(all_factors)})') from None

    logger.info(f'ALPHA_FACTORS_FILE|{alpha_factors_file_name}')
    all_factors.to_csv(alpha_factors_file_name)
    logger.info('Alpha factors saved.')

    for alpha_factor in all_factors.columns:
        logger.info(f'ALPHA_FACTOR|{alpha_factor}')

    # Generate Beta Factors
    logger.info(f'Generate beta factors...')
    returns = alpha_factors.FactorReturns(price_histories).factor_data.dropna()
    end_date = returns.index.max()
    number_of_beta_years = int(alpha_config["NumberOfYearsPriceHistories"]) - 1
    start_date = end_date - pd.offsets.DateOffset(years=number_of_beta_years)
    logger.info(f'Generating {number_of_beta_years} year Betas from {start_date} to {end_date}')
    beta_dates = pd.date_range(start_date, end_date, freq='D')
    daily_betas = {}
    for beta_date in tqdm(returns[start_date:].index, desc='Dates', unit=' Daily Beta'):
        start_of_returns = beta_date - pd.offsets.DateOffset(years=1)
        beta_returns = returns.loc[start_of_returns:beta_date]
        risk_model = alpha_factors.RiskModelPCA(beta_returns, 1, 20)
        daily_betas[beta_date.strftime('%m/%d/%Y')] = risk_model

    logger.info(f'BETA_FACTORS_FILE|{beta_factors_file_name}')
    with open(beta_factors_file_name, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(daily_betas, f, pickle.HIGHEST_PROTOCOL)


###########################################################
# Stand-alone execution
###########################################################

if __name__ == '__main__':
    generate_alpha_beta_factors()
