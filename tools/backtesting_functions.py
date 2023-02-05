import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import tools.trading_factors_yahoo as alpha_factors
from portfolio_optimizer import OptimalHoldings


def save_alpha_vectors(factors_df: pd.DataFrame, storage_path: Path = None):
    logger = logging.getLogger('AlphaFactorsHelper.save_alpha_factors')
    if storage_path is None:
        logger.info(f'ALPHA_VECTORS_FILE|NOT_SAVED')
        return

    storage_path.parent.mkdir(parents=True, exist_ok=True)
    if storage_path.suffix == 'parquet':
        factors_df.to_parquet(storage_path)
    else:
        factors_df.to_csv(storage_path, index=True)
    logger.info(f'ALPHA_VECTORS_FILE|SAVED|{storage_path}')


def load_alpha_vectors(storage_path: Path = None) -> pd.DataFrame:
    if storage_path.suffix == 'parquet':
        return pd.read_parquet(storage_path)
    else:
        return pd.read_csv(storage_path, parse_dates=['Date']).set_index(['Date']).sort_index()


def get_alpha_vectors(ai_alpha_factors: pd.DataFrame, storage_path: Path = None, reload: bool = False) -> pd.DataFrame:
    logger = logging.getLogger('BacktestingFunctions.get_alpha_vectors')
    logger.info('Getting Alpha Vectors...')
    if storage_path is not None and storage_path.exists():
        logger.info(f'ALPHA_VECTORS_FILE|EXISTS|{storage_path}')
        if not reload:
            logger.info(f'ALPHA_VECTORS_FILE|RELOAD|{reload}')
            return load_alpha_vectors(storage_path)

    ai_alpha_name = ai_alpha_factors.columns.to_list()[0]
    alpha_vectors = ai_alpha_factors.reset_index().pivot(index='Date', columns='Symbols', values=ai_alpha_name)
    save_alpha_vectors(alpha_vectors, storage_path)
    return alpha_vectors


def load_beta_factors(storage_path):
    return pickle.load(open(storage_path, 'rb'))


def save_beta_factors(daily_betas: dict, storage_path):
    logger = logging.getLogger('BacktestingFunctions.save_beta_factors')
    if storage_path is None:
        logger.info(f'BETA_FACTORS_FILE|NOT_SAVED')
        return

    storage_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'BETA_FACTORS_FILE|SAVED|{storage_path}')
    with open(storage_path, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(daily_betas, f, pickle.HIGHEST_PROTOCOL)


def generate_beta_factors(price_histories: pd.DataFrame, number_of_years: int = 2, num_exposures: int = 20,
                          storage_path: Path = None, reload: bool = False) -> dict:
    logger = logging.getLogger('BacktestingFunctions.generate_beta_factors')
    logger.info(f'Generate beta factors...')
    if storage_path is not None and storage_path.exists():
        logger.info(f'BETA_FACTORS_FILE|EXISTS|{storage_path}')
        if not reload:
            logger.info(f'BETA_FACTORS_FILE|RELOAD|{reload}')
            return load_beta_factors(storage_path)

    logger.info(f'NUMBER_OF_YEARS|{number_of_years}')
    logger.info(f'NUMBER_OF_EXPOSURES|{num_exposures}')
    returns = alpha_factors.FactorReturns(price_histories).factor_data.dropna()
    end_date = returns.index.max()
    number_of_beta_years = number_of_years - 1
    start_date = end_date - pd.offsets.DateOffset(years=number_of_beta_years)
    logger.info(f'Generating {number_of_beta_years} year Betas from {start_date} to {end_date}')
    daily_betas = {}
    for beta_date in tqdm(returns[start_date:].index, desc='Dates', unit=' Daily Beta'):
        start_of_returns = beta_date - pd.offsets.DateOffset(years=1)
        beta_returns = returns.loc[start_of_returns:beta_date]
        risk_model = alpha_factors.RiskModelPCA(beta_returns, 1, num_exposures)
        daily_betas[beta_date] = risk_model
    save_beta_factors(daily_betas, storage_path)
    return daily_betas


def get_lambda(average_dollar_volume):
    average_dollar_volume_cleaned = average_dollar_volume.replace(np.nan, 1.0e4)
    average_dollar_volume_cleaned = average_dollar_volume_cleaned.replace(0.0, 1.0e4)
    return 0.1 / average_dollar_volume_cleaned


def get_total_transaction_costs(h0, h_star, total_cost_lambda):
    return np.dot((h_star - h0) ** 2, total_cost_lambda)


def backtest_factors(price_histories: pd.DataFrame,
                     alpha_vectors: pd.DataFrame,
                     daily_betas: dict,
                     forward_prediction_days: int = 5,
                     backtest_days: int = 10,
                     risk_cap: float = 0.05,
                     weights_max: float = 0.10,
                     weights_min: float = 0.0,
                     ) -> (pd.Series, pd.DataFrame):
    logger = logging.getLogger('BacktestingFunctions.backtest_factors')
    logger.info('Running backtest...')
    returns = alpha_factors.FactorReturns(price_histories).factor_data
    delayed_returns = returns[-backtest_days:].shift(-forward_prediction_days).dropna()
    opt_dates = delayed_returns.index.to_list()
    optimal_holdings_df = predict_optimal_holdings(alpha_vectors,
                                                   daily_betas,
                                                   opt_dates,
                                                   risk_cap,
                                                   weights_max,
                                                   weights_min)

    adv = alpha_factors.AverageDollarVolume(price_histories, forward_prediction_days).factor_data
    tc_lambda = get_lambda(adv)
    current_holdings = pd.Series(np.zeros(len(delayed_returns.columns)), index=delayed_returns.columns)
    estimated_returns_by_date = {}
    for key, est_return in tqdm(delayed_returns.iterrows(), desc='Optimal Holdings', unit=' Date'):
        optimal_holdings = optimal_holdings_df.loc[key]
        returns_from_holdings = (optimal_holdings * est_return).sum()
        trading_costs = get_total_transaction_costs(current_holdings, optimal_holdings, tc_lambda.loc[key])
        estimated_returns_by_date[key] = returns_from_holdings + trading_costs
        current_holdings = optimal_holdings

    estimated_returns_by_date_se = pd.Series(estimated_returns_by_date.values(),
                                             index=estimated_returns_by_date.keys(),
                                             name='Returns')
    return estimated_returns_by_date_se, optimal_holdings_df


def predict_optimal_holdings(alpha_vectors: pd.DataFrame,
                             daily_betas: dict,
                             opt_dates: list = None,
                             risk_cap: float = 0.05,
                             weights_max: float = 0.10,
                             weights_min: float = 0.0,
                             ) -> pd.DataFrame:
    if opt_dates is None:
        opt_dates = list(daily_betas.keys())
    logger = logging.getLogger('BacktestingFunctions.predict_optimal_holdings')
    logger.info(f'PARAMETERS|risk_cap|{risk_cap}|weights_max|{weights_max}|weights_min|{weights_min}')
    logger.info(f'DATE_RANGE|{opt_dates[0]}|{opt_dates[-1]}')
    holdings_dict = {}
    for opt_date in tqdm(opt_dates, desc='Dates',
                         unit=' Portfolio Optimization'):
        alpha_vector = pd.DataFrame(alpha_vectors.loc[opt_date])
        risk_model = daily_betas[opt_date]
        optimal_weights = OptimalHoldings(risk_cap=risk_cap, weights_max=weights_max,
                                          weights_min=weights_min).find(alpha_vector,
                                                                        risk_model.factor_betas_,
                                                                        risk_model.factor_cov_matrix_,
                                                                        risk_model.idiosyncratic_var_vector_)
        holdings_dict[pd.to_datetime(opt_date)] = optimal_weights['optimalWeights']
    return pd.DataFrame.from_dict(holdings_dict, orient="index").round(2)
