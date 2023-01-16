import logging.config
from configparser import SectionProxy
from pathlib import Path
import tools.utils as utils
import pandas as pd
import yfinance as yf

DEFAULT_SNP500_FILE = 'snp500.csv'


def default_histories_path(configuration: SectionProxy) -> Path:
    file_name = configuration["DataDirectory"] + '/' + configuration["PriceHistoriesFileName"]
    file_path = Path(file_name)
    return file_path


def default_snp500_path_config(configuration: SectionProxy) -> Path:
    return default_snp500_path(default_histories_path(configuration))


# This will use the same directory as the histories' path by default
def default_snp500_path(histories_path: Path) -> Path:
    if histories_path is None:
        return None

    return Path(histories_path.parent, DEFAULT_SNP500_FILE)


def ensure_data_directory(storage_path: Path):
    if storage_path is None:
        return
    storage_path.parent.mkdir(parents=True, exist_ok=True)


def from_yahoo_finance_config(configuration: SectionProxy,
                              reload=False) -> pd.DataFrame:
    price_histories_path = default_histories_path(configuration)
    period = configuration.get("NumberOfYearsPriceHistories", '5') + 'y'
    return from_yahoo_finance(storage_path=price_histories_path, period=period, reload=reload)


def from_yahoo_finance(symbols: [] = [],
                       start=None, end=None, period='5y',
                       storage_path: Path = None,
                       reload=False) -> pd.DataFrame:
    logger = logging.getLogger('from_yahoo_finance')
    logger.info(f'Pandas version: {pd.__version__}')
    logger.info(f'Yahoo Finance version: {yf.__version__}')

    if storage_path is not None and storage_path.exists():
        logger.info(f'PRICE_HISTORIES_FILE|EXISTS')
        if not reload:
            logger.info(f'PRICE_HISTORIES_FILE|RELOAD|{reload}')
            return load_price_histories(storage_path)

    if len(symbols) == 0:
        snp500_path = default_snp500_path(storage_path)
        symbols = load_snp500_symbols(snp500_path, reload=True).index.to_list()
        logger.info(f'SYMBOLS|S&P500|{len(symbols)}')
    else:
        logger.info(f'SYMBOLS|CUSTOM|{len(symbols)}')

    price_histories = download_histories_and_adjust(symbols, start, end, period)
    save_price_histories(price_histories, storage_path)
    logger.info('Done gathering S&P 500 price histories...')
    return price_histories


def save_price_histories(price_histories: pd.DataFrame, storage_path: Path = None):
    logger = logging.getLogger('price_histories_helper.save')
    if storage_path is None:
        logger.info(f'PRICE_HISTORIES_FILE|NOT_SAVED')
        return

    ensure_data_directory(storage_path)
    price_histories.to_csv(storage_path, index=True)
    logger.info(f'PRICE_HISTORIES_FILE|SAVED|{storage_path}')


def load_price_histories(storage_path: Path = None) -> pd.DataFrame:
    return pd.read_csv(storage_path,
                       header=[0, 1], index_col=[0], parse_dates=True, low_memory=False)


def download_histories_and_adjust(symbols: [], start=None, end=None, period='5y') -> pd.DataFrame:
    logger = logging.getLogger('price_histories_helper.download')
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


def load_snp500_symbols(storage_path: Path, reload: bool = False) -> pd.DataFrame:
    logger = logging.getLogger('price_histories_helper.load_snp500')
    logger.info(f'SNP500_FILE|{storage_path}')
    if storage_path is not None and storage_path.exists():
        logger.info(f'SNP500_FILE|EXISTS')
        if not reload:
            logger.info(f'SNP500_FILE|RELOAD|{reload}')
            return pd.read_csv(storage_path, index_col=[0], low_memory=False)

    snp_500_stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                                  header=0,
                                  attrs={'id': 'constituents'},
                                  index_col='Symbol')[0]

    if storage_path is not None:
        ensure_data_directory(storage_path)
        snp_500_stocks.to_csv(storage_path)

    return snp_500_stocks


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s|%(name)s|%(levelname)s|%(message)s', level=logging.INFO)
    logging.root = logging.getLogger('price_histories_helper')
    from_yahoo_finance(symbols=['AAPL', 'GOOG'])
