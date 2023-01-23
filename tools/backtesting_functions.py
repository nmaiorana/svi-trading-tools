import logging
import pickle
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import tools.trading_factors_yahoo as alpha_factors


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

    logger.info(f'BETA_FACTORS_FILE|SAVED|{storage_path}')
    with open(storage_path, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(daily_betas, f, pickle.HIGHEST_PROTOCOL)


def generate_beta_factors(price_histories: pd.DataFrame, number_of_years: int = 2,
                          storage_path: Path = None, reload: bool = False) -> dict:
    logger = logging.getLogger('BacktestingFunctions.generate_beta_factors')
    logger.info(f'Generate beta factors...')
    if storage_path is not None and storage_path.exists():
        logger.info(f'BETA_FACTORS_FILE|EXISTS|{storage_path}')
        if not reload:
            logger.info(f'BETA_FACTORS_FILE|RELOAD|{reload}')
            return load_beta_factors(storage_path)

    returns = alpha_factors.FactorReturns(price_histories).factor_data.dropna()
    end_date = returns.index.max()
    number_of_beta_years = number_of_years - 1
    start_date = end_date - pd.offsets.DateOffset(years=number_of_beta_years)
    logger.info(f'Generating {number_of_beta_years} year Betas from {start_date} to {end_date}')
    beta_dates = pd.date_range(start_date, end_date, freq='D')
    daily_betas = {}
    for beta_date in tqdm(returns[start_date:].index, desc='Dates', unit=' Daily Beta'):
        start_of_returns = beta_date - pd.offsets.DateOffset(years=1)
        beta_returns = returns.loc[start_of_returns:beta_date]
        risk_model = alpha_factors.RiskModelPCA(beta_returns, 1, 20)
        daily_betas[beta_date.strftime('%m/%d/%Y')] = risk_model
    save_beta_factors(daily_betas, storage_path)
    return daily_betas

