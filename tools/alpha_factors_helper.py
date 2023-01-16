import logging
from configparser import SectionProxy
import pandas as pd

import tools.trading_factors_yahoo as alpha_factors
import tools.price_histories_helper as phh
import tools.utils as utils

logging.basicConfig(format='%(asctime)s|%(name)s|%(levelname)s|%(message)s', level=logging.INFO)
logger = logging.getLogger('AlphaFactorsHelper')


def eval_factor_and_add(factors_list, factors_to_use,
                        factor: alpha_factors.FactorData,
                        pricing: pd.DataFrame,
                        min_sharpe_ratio=0.5):
    logger.info(f'FACTOR_EVAL|{factor.factor_name}|{min_sharpe_ratio}...')
    factor_data = factor.for_al()
    clean_factor_data, unix_time_factor_data = alpha_factors.prepare_alpha_lens_factor_data(
        factor_data.to_frame().copy(),
        pricing)
    factor_returns = alpha_factors.get_factor_returns(clean_factor_data)
    sharpe_ratio = alpha_factors.compute_sharpe_ratio(factor_returns)['Sharpe Ratio'].values[0]
    factors_list.append(factor_data)
    if sharpe_ratio < min_sharpe_ratio:
        logger.info(f'FACTOR_EVAL|{factor.factor_name}|{min_sharpe_ratio}|{sharpe_ratio}|REJECTED')
        return
    logger.info(f'FACTOR_EVAL|{factor.factor_name}|{min_sharpe_ratio}|{sharpe_ratio}|ACCEPTED')
    factors_to_use.append(factor.factor_name)


def generate_alpha_factors(configuration: SectionProxy):
    price_histories = phh.from_yahoo_finance_config(configuration, reload=False)
    close = price_histories.Close
    logger.info(f'STOCK_TICKERS|{len(close.columns)}')
    alpha_factors_file_name = configuration["DataDirectory"] + '/' + configuration["AlphaFactorsFileName"]

    logger.info('Gathering stock ticker sector data...')
    snp_500_stocks = utils.get_snp500()
    sector_helper = alpha_factors.get_sector_helper(snp_500_stocks, 'GICS Sector', close.columns)
    logger.info(f'Stock sector information gathered.')
    alpha_factors_list = []
    selected_factor_names = []

    min_sharpe_ratio = float(alpha_config['min_sharpe_ratio'])
    logger.info(f'FACTOR_EVAL|MIN_SHARPE_RATIO|{min_sharpe_ratio}')

    potential_factors = [
        alpha_factors.FactorMomentum(price_histories, 252).demean(group_by=sector_helper.values()).rank().zscore(),
        alpha_factors.TrailingOvernightReturns(price_histories, 10).rank().zscore().smoothed(10).rank().zscore(),
        alpha_factors.FactorMeanReversion(price_histories, 30).demean(group_by=sector_helper.values()).rank().zscore(),
        alpha_factors.FactorMeanReversion(price_histories, 60).demean(group_by=sector_helper.values()).rank().zscore(),
        alpha_factors.FactorMeanReversion(price_histories, 90).demean(group_by=sector_helper.values()).rank().zscore(),
        alpha_factors.AnnualizedVolatility(price_histories, 20).rank().zscore(),
        alpha_factors.AnnualizedVolatility(price_histories, 120).rank().zscore(),
        alpha_factors.AverageDollarVolume(price_histories, 20).rank().zscore(),
        alpha_factors.AverageDollarVolume(price_histories, 120).rank().zscore(),
    ]
    for potential_factor in potential_factors:
        afh.eval_factor_and_add(alpha_factors_list, selected_factor_names, potential_factor, close, min_sharpe_ratio)

    # Fixed Factors
    market_dispersion = alpha_factors.MarketDispersion(price_histories, 120)
    alpha_factors_list.append(market_dispersion.for_al())
    selected_factor_names.append(market_dispersion.factor_name)
    market_volatility = alpha_factors.MarketVolatility(price_histories, 120)
    alpha_factors_list.append(market_volatility.for_al())
    selected_factor_names.append(market_volatility.factor_name)
