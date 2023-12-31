import logging
from pathlib import Path
import pandas as pd
import pickle
from alphalens.utils import MaxLossExceededError
from sklearn.ensemble import RandomForestClassifier
import trading_factors_yahoo as alpha_factors
from nonoverlapping_estimator import NoOverlapVoter
import matplotlib.pyplot as plt

plt.interactive(False)


# TODO: Add standard factors
# TODO: Save Factors
# TODO: Create Configuration for factors used for model training features

def eval_factor(factor_data: pd.Series,
                price_histories: pd.DataFrame,
                min_sharpe_ratio=0.5) -> bool:
    logger = logging.getLogger('AlphaFactorsHelper.eval_factor')
    logger.info(
        f'Evaluate factor {factor_data.name}({len(factor_data)}) with a minimum Sharpe Ratio of {min_sharpe_ratio}...')

    try:
        clean_factor_data, unix_time_factor_data = alpha_factors.prepare_alpha_lens_factor_data(
            factor_data.to_frame().copy(),
            price_histories.Close)
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


def identify_factors_to_use(factors_df: pd.DataFrame, price_histories: pd.DataFrame, min_sharpe_ratio=0.5) -> list:
    factors_to_use = []
    for factor_name in factors_df.columns:
        use_factor = eval_factor(factors_df[factor_name], price_histories, min_sharpe_ratio)
        if use_factor:
            factors_to_use.append(factor_name)
    return factors_to_use


def get_sector_helper(snp_500_stocks: pd.DataFrame, price_histories: pd.DataFrame) -> dict:
    logger = logging.getLogger('AlphaFactorsHelper.sector_helper')
    logger.info('Gathering stock ticker sector data...')
    sector_helper = alpha_factors.get_sector_helper(snp_500_stocks, 'GICS Sector', price_histories.Close.columns)
    logger.info(f'Stock sector information gathered.')
    return sector_helper


def save_alpha_factors(factors_df: pd.DataFrame, storage_path: Path = None):
    logger = logging.getLogger('AlphaFactorsHelper.save_alpha_factors')
    if storage_path is None:
        logger.info(f'ALPHA_FACTORS_FILE|NOT_SAVED')
        return

    storage_path.parent.mkdir(parents=True, exist_ok=True)
    if storage_path.suffix == '.parquet':
        factors_df.to_parquet(storage_path)
    else:
        factors_df.to_csv(storage_path)
    logger.info(f'ALPHA_FACTORS_FILE|SAVED|{storage_path}')


def load_alpha_factors(storage_path: Path = None) -> pd.DataFrame:
    if storage_path.suffix == '.parquet':
        return pd.read_parquet(storage_path)
    else:
        return pd.read_csv(storage_path, parse_dates=['Date']).set_index(['Date', 'Symbols']).sort_index()


def get_alpha_factors(price_histories: pd.DataFrame = None, sector_helper: dict = None, factors_array: list = None,
                      storage_path: Path = None, reload: bool = False):
    logger = logging.getLogger('AlphaFactorsHelper.get_alpha_factors')
    if storage_path is not None and storage_path.exists():
        logger.info(f'ALPHA_FACTORS_FILE|EXISTS|{storage_path}')
        if not reload:
            logger.info(f'ALPHA_FACTORS_FILE|RELOAD|{reload}')
            return load_alpha_factors(storage_path)

    alpha_factors_df = generate_factors_df(price_histories, sector_helper, factors_array)
    save_alpha_factors(alpha_factors_df, storage_path)
    return alpha_factors_df


def generate_factors_df(price_histories: pd.DataFrame = None,
                        sector_helper: dict = None,
                        factors_array: list = None,
                        ) -> pd.DataFrame:
    logger = logging.getLogger('AlphaFactorsHelper.generate_factors_df')
    if price_histories is not None and sector_helper is None:
        logger.error('You have to define sector_helper if using price_histories!' +
                     'Are you trying to pass factors_array? Use factors_array = array.')
        raise ValueError('You have to define sector_helper if using price_histories!')
    logger.info(f'Generating factors...')
    if factors_array is None:
        factors_array = default_factors(price_histories, sector_helper)
    factors_df = pd.concat(factors_array, axis=1)
    factors_df.reset_index(inplace=True)
    factors_df['Date'] = pd.to_datetime(factors_df['Date'])
    factors_df.set_index(['Date', 'Symbols'], inplace=True)

    # Date Factors
    logger.info(f'Adding date parts...')
    alpha_factors.FactorDateParts(factors_df)
    logger.info(f'Done generating factors.')
    return factors_df.dropna()


def default_factors(price_histories: pd.DataFrame, sector_helper: dict) -> list:
    """
    Using a history of prices, generate a list of standard factors to use for training.

    :rtype: list
    """
    factors_array = [
        alpha_factors.FactorMomentum(price_histories, 252).demean(
            group_by=sector_helper.values()).rank().zscore().for_al(),
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
    return factors_array


def load_ai_alpha_model(storage_path):
    with open(storage_path, 'rb') as f:
        return pickle.load(f)


def get_ai_alpha_model(alpha_factors_df: pd.DataFrame,
                       price_histories: pd.DataFrame,
                       min_sharpe_ratio: float = 0.85,
                       forward_prediction_days: int = 5,
                       target_quantiles: int = 2,
                       n_trees: int = 50,
                       storage_path: Path = None,
                       reload: bool = False) -> NoOverlapVoter:
    logger = logging.getLogger('AlphaFactorsHelper.get_ai_alpha_model')
    if storage_path is not None and storage_path.exists():
        logger.info(f'AI_ALPHA_MODEL_FILE|EXISTS|{storage_path}')
        if not reload:
            logger.info(f'AI_ALPHA_MODEL_FILE|RELOAD|{reload}')
            return load_ai_alpha_model(storage_path)

    ai_alpha_model = train_ai_alpha_model(alpha_factors_df,
                                          price_histories,
                                          min_sharpe_ratio,
                                          forward_prediction_days,
                                          target_quantiles,
                                          n_trees)
    if storage_path is None:
        logger.info(f'AI_ALPHA_MODEL_FILE|NOT_SAVED')
    else:
        logger.info(f'AI_ALPHA_MODEL_FILE|SAVED|{storage_path}')
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(storage_path, 'wb') as f:
            pickle.dump(ai_alpha_model, f, pickle.HIGHEST_PROTOCOL)
    return ai_alpha_model


def train_ai_alpha_model(alpha_factors_df: pd.DataFrame,
                         price_histories: pd.DataFrame,
                         min_sharpe_ratio: float = 0.85,
                         forward_prediction_days: int = 5,
                         target_quantiles: int = 2,
                         n_trees: int = 50) -> NoOverlapVoter:
    # - Compute target values (y)
    #     - Quantize with 2 bins
    # - Train model for Feature importance
    # - Feature reduction
    # - Train model for AI Alpha Vector
    # - Compute AI Alpha Vectors for 1 year
    # - Save AI Alpha Vectors

    # ## Compute the target values (y) and Shift back to create a 5-day forward prediction
    #
    # This is something you want to experiment with. If you are planning on holding on to assets for long periods of
    # time, perhaps a 20, 40 or 60 forward prediction will work better.

    logger = logging.getLogger('AlphaFactorsHelper.train_ai_alpha_model')
    logger.info(f'Training ai alpha model')
    prod_target_source = f'{forward_prediction_days}Day{target_quantiles}Quant'
    logger.info(
        f'Setting {forward_prediction_days} days-{target_quantiles} quantiles to target {prod_target_source}')
    logger.info(f'Factors from date: {alpha_factors_df.index.get_level_values("Date").min()}' +
                f'to date: {alpha_factors_df.index.get_level_values("Date").max()}')

    features = identify_factors_to_use(alpha_factors_df, price_histories, min_sharpe_ratio)
    training_factors = pd.concat(
        [
            alpha_factors_df[features],
            alpha_factors.FactorReturnQuantiles(
                price_histories, target_quantiles, forward_prediction_days).for_al(prod_target_source),
        ], axis=1).dropna()
    training_factors.sort_index(inplace=True)

    training_factors['target'] = training_factors.groupby(
        level='Symbols')[prod_target_source].shift(-forward_prediction_days)

    logger.info(f'Creating training and label data...')
    temp = training_factors.dropna().copy()
    X = temp[features]
    y = temp['target']
    for feature in features:
        logger.info(f'TRAINING_FEATURE|{feature}')

    logger.info(f'TRAINING_DATASET|{len(X)}|LABEL_DATASET|{len(y)}')

    n_stocks = len(set(alpha_factors_df.index.get_level_values(level='Symbols').values))

    clf_parameters = {
        'criterion': 'entropy',
        'min_samples_leaf': forward_prediction_days * n_stocks,
        'oob_score': True,
        'n_jobs': -1,
        'random_state': 42}

    logger.info(f'Creating RandomForestClassifier with {n_trees} trees...')
    for key, value in clf_parameters.items():
        logger.info(f'Parameter: {key} set to {value}')
    clf = RandomForestClassifier(n_trees, **clf_parameters)

    logger.info(f'Creating Non-Overlapping Voter with {forward_prediction_days - 1} non-overlapping windows...')
    clf_nov = NoOverlapVoter(clf, n_skip_samples=forward_prediction_days - 1)

    logger.info(f'Training classifier...')
    clf_nov.fit(X, y)

    logger.info(f'CLASSIFIER|TRAIN_SCORE|{clf_nov.score(X, y.values)}|OOB_SCORE|{clf_nov.oob_score_}')
    return clf_nov


def get_ai_alpha_factor(alpha_factors_df: pd.DataFrame,
                        ai_alpha_model: NoOverlapVoter,
                        ai_alpha_name: str = 'AI_ALPHA',
                        storage_path: Path = None,
                        reload: bool = False):
    logger = logging.getLogger('AlphaFactorsHelper.get_ai_alpha_factor')
    logger.info(f'Generating AI Alpha Score...')
    if storage_path is not None and storage_path.exists():
        logger.info(f'AI_ALPHA_VECTOR_FILE|EXISTS|{storage_path}')
        if not reload:
            logger.info(f'AI_ALPHA_VECTOR_FILE|RELOAD|{reload}')
            return load_alpha_factors(storage_path)

    factors_with_alpha = alpha_factors.add_alpha_score(alpha_factors_df[ai_alpha_model.feature_names_in_],
                                                       ai_alpha_model,
                                                       ai_alpha_name)
    ai_alpha_factor_df = factors_with_alpha[ai_alpha_name].to_frame()
    save_alpha_factors(ai_alpha_factor_df, storage_path)
    logger.info(f'Done Generating AI Alpha ({len(ai_alpha_factor_df)}).')
    return ai_alpha_factor_df

