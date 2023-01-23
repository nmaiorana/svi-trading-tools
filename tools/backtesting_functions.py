# TODO: Create a series of functions to help in backtesting. But first, create tests for alpha/beta
import logging
from pathlib import Path

import pandas as pd


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


def get_alpha_vectors(ai_alpha_factors: pd.DataFrame, storage_path: Path = None, reload: bool = False):
    logger = logging.getLogger('BacktestingFunctions.get_alpha_vectors')
    logger.info('Getting Alpha Vectors')
    if storage_path is not None and storage_path.exists():
        logger.info(f'ALPHA_VECTORS_FILE|EXISTS|{storage_path}')
        if not reload:
            logger.info(f'ALPHA_VECTORS_FILE|RELOAD|{reload}')
            return load_alpha_vectors(storage_path)

    alpha_vectors = ai_alpha_factors.reset_index().pivot(index='Date', columns='Symbols')
    save_alpha_vectors(alpha_vectors, storage_path)
    return alpha_vectors
