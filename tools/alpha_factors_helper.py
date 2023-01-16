import logging
from configparser import SectionProxy
import pandas as pd
from alphalens.utils import MaxLossExceededError

import tools.trading_factors_yahoo as alpha_factors
import tools.price_histories_helper as phh

logging.basicConfig(format='%(asctime)s|%(name)s|%(levelname)s|%(message)s', level=logging.INFO)


# TODO: Add standard factors
# TODO: Save Factors
# TODO: Create Configuration for factors used for model training features

def eval_factor(factor_data: pd.Series,
                pricing: pd.DataFrame,
                min_sharpe_ratio=0.5) -> bool:
    logger = logging.getLogger('AlphaFactorsHelper.eval_factor')
    logger.info(f'Evaluate factor {factor_data.name} with a minimum Sharpe Ratio of {min_sharpe_ratio}...')

    try:
        clean_factor_data, unix_time_factor_data = alpha_factors.prepare_alpha_lens_factor_data(
            factor_data.to_frame().copy(),
            pricing)
        factor_returns = alpha_factors.get_factor_returns(clean_factor_data)
        sharpe_ratio = alpha_factors.compute_sharpe_ratio(factor_returns)['Sharpe Ratio'].values[0]
    except MaxLossExceededError:
        logger.info(f'FACTOR_EVAL|{factor_data.name}|ACCEPTED|MAX_LOSS_EXCEEDED')
        return True

    if sharpe_ratio < min_sharpe_ratio:
        logger.info(f'FACTOR_EVAL|{factor_data.name}|{min_sharpe_ratio}|{sharpe_ratio}|REJECTED')
        return False

    logger.info(f'FACTOR_EVAL|{factor_data.name}|{min_sharpe_ratio}|{sharpe_ratio}|ACCEPTED')
    return True


def identify_factors_to_use(factors_df: pd.DataFrame, pricing: pd.DataFrame, min_sharpe_ratio=0.5) -> []:
    factors_to_use = []
    for factor_name in factors_df.columns:
        use_factor = eval_factor(factors_df[factor_name], pricing, min_sharpe_ratio)
        if use_factor:
            factors_to_use.append(factor_name)
    return factors_to_use


def get_sector_helper(configuration: SectionProxy, close: pd.DataFrame) -> dict:
    logger = logging.getLogger('AlphaFactorsHelper.sector_helper')
    logger.info('Gathering stock ticker sector data...')
    snp_500_stocks = phh.load_snp500_symbols(phh.default_snp500_path_config(configuration))
    sector_helper = alpha_factors.get_sector_helper(snp_500_stocks, 'GICS Sector', close.columns)
    logger.info(f'Stock sector information gathered.')
    return sector_helper


def generate_factors(price_histories: pd.DataFrame, sector_helper: dict) -> pd.DataFrame:
    logger = logging.getLogger('AlphaFactorsHelper.scored_factors')
    logger.info(f'Generating factors...')
    factors_array = [
        alpha_factors.FactorMomentum(price_histories, 252)
        .demean(group_by=sector_helper.values()).rank().zscore().for_al(),
        alpha_factors.TrailingOvernightReturns(price_histories, 10)
        .rank().zscore().smoothed(10).rank().zscore().for_al(),
        alpha_factors.FactorMeanReversion(price_histories, 30).demean(
            group_by=sector_helper.values()).rank().zscore().for_al(),
        alpha_factors.FactorMeanReversion(price_histories, 60).demean(
            group_by=sector_helper.values()).rank().zscore().for_al(),
        alpha_factors.FactorMeanReversion(price_histories, 90).demean(
            group_by=sector_helper.values()).rank().zscore().for_al(),
        alpha_factors.AnnualizedVolatility(price_histories, 20).rank().zscore().for_al(),
        alpha_factors.AnnualizedVolatility(price_histories, 120).rank().zscore().for_al(),
        alpha_factors.AverageDollarVolume(price_histories, 20).rank().zscore().for_al(),
        alpha_factors.AverageDollarVolume(price_histories, 120).rank().zscore().for_al(),
        # Regime factors
        alpha_factors.MarketDispersion(price_histories, 120).for_al(),
        alpha_factors.MarketVolatility(price_histories, 120).for_al()
    ]
    factors_df = pd.concat(factors_array, axis=1)
    # Date Factors
    logger.info(f'Adding date parts...')
    alpha_factors.FactorDateParts(factors_df)
    logger.info(f'Done generating factors.')
    return factors_df.dropna()


