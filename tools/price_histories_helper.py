import logging.config
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf


def ensure_data_directory(storage_path: Path):
    if storage_path is None:
        return
    storage_path.parent.mkdir(parents=True, exist_ok=True)


def from_yahoo_finance(symbols: list,
                       start=None, end=None, period='5y',
                       storage_path: Path = None,
                       reload=False) -> pd.DataFrame:
    logger = logging.getLogger('phh.from_yahoo_finance')
    logger.info(f'Pandas version: {pd.__version__}')
    logger.info(f'Yahoo Finance version: {yf.__version__}')

    if storage_path is not None and storage_path.exists():
        logger.info(f'PRICE_HISTORIES_FILE|EXISTS|{storage_path}')
        if not reload:
            logger.info(f'PRICE_HISTORIES_FILE|RELOAD|{reload}')
            return load_price_histories(storage_path)

    price_histories = download_histories_and_adjust(symbols, start, end, period)
    save_price_histories(price_histories, storage_path)
    logger.info(f'Done gathering {len(symbols)} price histories...')
    return price_histories


def save_price_histories(price_histories: pd.DataFrame, storage_path: Path = None):
    logger = logging.getLogger('phh.save_price_histories')
    if storage_path is None:
        logger.info(f'PRICE_HISTORIES_FILE|NOT_SAVED')
        return

    ensure_data_directory(storage_path)
    if storage_path.suffix == '.parquet':
        price_histories.to_parquet(storage_path, index=True)
    else:
        price_histories.to_csv(storage_path, index=True)
    logger.info(f'PRICE_HISTORIES_FILE|SAVED|{storage_path}')


def load_price_histories(storage_path: Path = None) -> pd.DataFrame:
    logger = logging.getLogger('phh.load_price_histories')
    if storage_path.suffix == '.parquet':
        price_histories = pd.read_parquet(storage_path)
    else:
        price_histories = pd.read_csv(storage_path,
                                      header=[0, 1], index_col=[0], parse_dates=True, low_memory=False)
    logger.info(f'PRICE_HISTORIES|{len(price_histories)}|{price_histories.index.min()}|{price_histories.index.max()}')
    return price_histories


def download_histories_and_adjust(symbols: list, start=None, end=None, period='5y') -> pd.DataFrame:
    logger = logging.getLogger('phh.download')
    if start is not None or end is not None:
        argv = {'start': start, 'end': end}
    else:
        argv = {'period': period}

    price_histories = yf.download(tickers=symbols, auto_adjust=True, **argv)
    price_histories.index = pd.DatetimeIndex(price_histories.index)
    price_histories.rename_axis(columns=['Attributes', 'Symbols'], inplace=True)
    price_histories = price_histories.round(2)
    logger.info(f'PRICE_HISTORIES|Back filling from first price for each symbol...')
    price_histories.bfill(inplace=True)
    logger.info(f'PRICE_HISTORIES|Forward filling from last price for each symbol...')
    price_histories.ffill(inplace=True)
    logger.info(f'PRICE_HISTORIES|Dropping any symbol with NA...')
    price_histories.dropna(axis=1, inplace=True)
    for dropped_symbol in (set(symbols) - set(price_histories.columns.get_level_values(1))):
        logger.warning(f'PRICE_HISTORIES|DROPPED|{dropped_symbol}')
    logger.info(f'PRICE_HISTORIES|{len(price_histories)}|{price_histories.index.min()}|{price_histories.index.max()}')
    return price_histories


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s|%(name)s|%(levelname)s|%(message)s', level=logging.INFO)
    logging.root = logging.getLogger('price_histories_helper')
    from_yahoo_finance(symbols=['AAPL', 'GOOG'])
