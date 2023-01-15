import pandas
import logging

import tools.trading_factors_yahoo as alpha_factors

logger = logging.getLogger('AlphaFactorsHelper')


def eval_factor_and_add(factors_list, factors_to_use,
                        factor: alpha_factors.FactorData,
                        pricing: pandas.DataFrame,
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
