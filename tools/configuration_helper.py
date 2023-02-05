from configparser import SectionProxy
from pathlib import Path

LONG_TERM_ASSET_TYPES = 'long_term_asset_types'

STRATEGY_CONFIG_FILENAME = 'StrategyConfigFileName'

LONG_TERM_STOCKS = 'long_term_stocks'
MASKED_ACCOUNT_NUMBER = 'masked_account_number'
ACCOUNTS = 'Accounts'
NUMBER_OF_RISK_EXPOSURES = 'NumberOfRiskExposures'
NAME = 'name'
FINAL_NAME = 'final_name'
AI_ALPHA_FILE_NAME = 'AIAlphaFileName'
MODEL_FILE_NAME = 'ModelFileName'
FACTORS_TO_USE_FILE_NAME = 'FactorsToUseFileName'
YEARS_PRICE_HISTORIES = 'NumberOfYearsPriceHistories'
DATA_DIRECTORY = "DataDirectory"
HISTORIES_FILE_NAME = "PriceHistoriesFileName"
ALPHA_FACTORS_FILE_NAME = "AlphaFactorsFileName"
ALPHA_VECTORS_FILE_NAME = "AlphaVectorsFileName"
BETA_FACTORS_FILE_NAME = "BetaFactorsFileName"


def get_data_directory(configuration: SectionProxy) -> Path:
    return Path(configuration[DATA_DIRECTORY])


def get_price_histories_path(configuration: SectionProxy) -> Path:
    return get_data_directory(configuration).joinpath(configuration.get(HISTORIES_FILE_NAME))


def get_alpha_factors_path(configuration: SectionProxy) -> Path:
    return get_data_directory(configuration).joinpath(configuration.get(ALPHA_FACTORS_FILE_NAME))


def get_number_of_years_of_price_histories_int(configuration: SectionProxy) -> int:
    return configuration.getint(YEARS_PRICE_HISTORIES)


def get_number_of_years_of_price_histories(configuration: SectionProxy) -> str:
    return configuration.get(YEARS_PRICE_HISTORIES) + 'y'


def get_strategy_path(configuration: SectionProxy):
    return get_data_directory(configuration).joinpath(configuration.get(NAME))


# TODO: Final version of all strategy paths
def get_ai_model_path(configuration: SectionProxy):
    return get_strategy_path(configuration).joinpath(configuration.get(MODEL_FILE_NAME))


def get_ai_alpha_path(configuration: SectionProxy):
    return get_strategy_path(configuration).joinpath(configuration.get(AI_ALPHA_FILE_NAME))


def get_alpha_vectors_path(configuration: SectionProxy):
    return get_strategy_path(configuration).joinpath(configuration.get(ALPHA_VECTORS_FILE_NAME))


def get_daily_betas_path(configuration: SectionProxy):
    return get_strategy_path(configuration).joinpath(configuration.get(BETA_FACTORS_FILE_NAME))


def get_final_strategy_path(configuration: SectionProxy):
    return get_data_directory(configuration).joinpath(configuration.get(FINAL_NAME))


def get_number_of_risk_exposures(strategy_config: SectionProxy) -> int:
    return strategy_config.getint(NUMBER_OF_RISK_EXPOSURES, 20)


def get_accounts(configuration: SectionProxy) -> list:
    return configuration.get(ACCOUNTS).split()


def get_masked_account_number(account_config: SectionProxy) -> str:
    return account_config.get(MASKED_ACCOUNT_NUMBER)


def get_long_term_stocks(account_config: SectionProxy) -> list:
    return account_config.get(LONG_TERM_STOCKS).split()


def get_long_term_asset_types(account_config: SectionProxy) -> list:
    return account_config.get(LONG_TERM_ASSET_TYPES).split()


def get_strategy_config_path(configuration: SectionProxy):
    return get_strategy_path(configuration).joinpath(configuration.get(STRATEGY_CONFIG_FILENAME, 'config.ini'))


def get_implemented_strategy(account_config: SectionProxy):
    return account_config.get('implemented_strategy')


def get_ai_model_final_path(strategy_config: SectionProxy):
    return get_final_strategy_path(strategy_config).joinpath(strategy_config.get(MODEL_FILE_NAME))


def get_ai_alpha_final_path(strategy_config: SectionProxy):
    return get_final_strategy_path(strategy_config).joinpath(strategy_config.get(AI_ALPHA_FILE_NAME))


def get_alpha_vectors_final_path(strategy_config: SectionProxy):
    return get_final_strategy_path(strategy_config).joinpath(strategy_config.get(ALPHA_VECTORS_FILE_NAME))


def get_daily_betas_final_path(strategy_config):
    return get_final_strategy_path(strategy_config).joinpath(strategy_config.get(BETA_FACTORS_FILE_NAME))