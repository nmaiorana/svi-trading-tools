from configparser import SectionProxy
from pathlib import Path

YEARS_PRICE_HISTORIES = 'NumberOfYearsPriceHistories'
DATA_DIRECTORY = "DataDirectory"
HISTORIES_FILE_NAME = "PriceHistoriesFileName"
ALPHA_FACTORS_FILE_NAME = "AlphaFactorsFileName"


def get_data_directory(configuration: SectionProxy) -> Path:
    return Path(configuration[DATA_DIRECTORY])


def get_price_histories_path(configuration: SectionProxy) -> Path:
    return get_data_directory(configuration).joinpath(configuration[HISTORIES_FILE_NAME])


def get_alpha_factors_path(configuration: SectionProxy) -> Path:
    return get_data_directory(configuration).joinpath(configuration[ALPHA_FACTORS_FILE_NAME])


def get_number_of_years_of_price_histories(configuration: SectionProxy) -> str:
    return configuration[YEARS_PRICE_HISTORIES] + 'y'
