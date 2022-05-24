# # Trading Functions
# 
# These functions were derived from an AI for Trading course provided by Udacity. The functions were originally part
# of quizzes / exercises given during the course.

import logging
import alphalens as al
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA


class FactorData:
    def __init__(self, factor_data_df, factor_name='Alpha Factor'):

        self.factor_data = factor_data_df
        self.factor_name = factor_name

    def compute(self):
        pass

    def top(self, n=10):
        return self.factor_data.iloc[-1].nlargest(n).index.tolist()

    def demean(self, groupby=None):
        if groupby is None:
            return FactorData(self.factor_data.sub(self.factor_data.mean(axis=1), axis=0), self.factor_name)

        demeaned_sectors = []
        for sector_tickers in groupby:
            sector_factor_data = self.factor_data[sector_tickers]
            demeaned_sectors.append(sector_factor_data.sub(sector_factor_data.mean(axis=1), axis=0))

        return FactorData(pd.concat(demeaned_sectors, axis=1).sort_index(axis=1), self.factor_name)

    def rank(self):
        return FactorData(self.factor_data.rank(axis=1), self.factor_name)

    def zscore(self):
        zscored_df = FactorData(
            pd.DataFrame(stats.zscore(self.factor_data, axis=1, nan_policy='omit'), index=self.factor_data.index,
                         columns=self.factor_data.columns), self.factor_name)
        zscored_df.factor_data.fillna(0, inplace=True)
        return zscored_df

    def smoothed(self, days=20):
        return FactorData(self.factor_data.rolling(window=days).mean(), self.factor_name + '_smoothed')

    def for_al(self, factor_name=None):
        if factor_name is not None:
            self.factor_name = factor_name
        alpha_lens_series = self.factor_data.stack()
        alpha_lens_series.name = self.factor_name
        return alpha_lens_series


class OpenPrices(FactorData):
    def __init__(self, price_histories_df, open_col='Open', close_col='Close'):
        self.open_col = open_col
        self.close_col = close_col
        super().__init__(factor_data_df=self.compute(price_histories_df), factor_name='open')

    def compute(self, price_histories_df):
        open_values = price_histories_df[self.open_col]
        close_values = price_histories_df[self.close_col]
        open_values = open_values.fillna(close_values.ffill())
        open_values = open_values.fillna(close_values.bfill())
        return open_values


class ClosePrices(FactorData):
    def __init__(self, price_histories_df, open_col='Open', close_col='Close'):
        self.open_col = open_col
        self.close_col = close_col
        super().__init__(factor_data_df=self.compute(price_histories_df), factor_name='close')

    def compute(self, price_histories_df):
        open_values = price_histories_df[self.open_col]
        close_values = price_histories_df[self.close_col]
        close_values = close_values.fillna(open_values.ffill())
        close_values = close_values.fillna(open_values.bfill())
        return close_values


class Volume(FactorData):
    def __init__(self, price_histories_df, volume_col='Volume'):
        self.volume_col = volume_col
        super().__init__(factor_data_df=self.compute(price_histories_df), factor_name='volume')

    def compute(self, price_histories_df):
        volume_values = price_histories_df[self.volume_col]
        volume_values = volume_values.fillna(volume_values.ffill())
        return volume_values


class DailyDollarVolume(FactorData):
    def __init__(self, price_histories_df):
        super().__init__(self.compute(price_histories_df), 'daily_dollar_volume')

    def compute(self, price_histories_df):
        return ClosePrices(price_histories_df).factor_data * Volume(price_histories_df).factor_data


class FactorReturns(FactorData):
    def __init__(self, price_histories_df, days=1):
        super().__init__(self.compute(price_histories_df, days), f'returns_{days}_day')

    def compute(self, price_histories_df, days):
        return ClosePrices(price_histories_df).factor_data.pct_change(days)


class FactorMomentum(FactorData):
    def __init__(self, price_histories_df, days=252):
        super().__init__(self.compute(price_histories_df, days), f'momentum_{days}_day')

    def compute(self, price_histories_df, days):
        return FactorReturns(price_histories_df, days).factor_data


class FactorMeanReversion(FactorData):
    def __init__(self, price_histories_df, days=5):
        super().__init__(self.compute(price_histories_df, days), f'mean_reversion_{days}_day_logret')

    def compute(self, price_histories_df, days):
        # Note, The idea is that we are looking for underperformed to revert towards the mean of the sector.
        #       Since the ranking will sort from under performers to over performers, we reverse the factor value by
        #       multiplying by -1, so that the largest underperformed are ranked higher.
        return -FactorReturns(price_histories_df, days).factor_data


class CloseToOpen(FactorData):
    def __init__(self, price_histories_df):
        super().__init__(self.compute(price_histories_df), f'close_to_open')

    def compute(self, price_histories_df):
        close_prices = ClosePrices(price_histories_df).factor_data
        open_prices = OpenPrices(price_histories_df).factor_data
        return (open_prices.shift(-1) - close_prices) / close_prices


class TrailingOvernightReturns(FactorData):
    def __init__(self, price_histories_df, days=5):
        super().__init__(self.compute(price_histories_df, days), f'trailing_overnight_returns_{days}_day')

    def compute(self, price_histories_df, days):
        return CloseToOpen(price_histories_df).factor_data.rolling(window=days, min_periods=1).sum()


# Universal Quant Features
class AnnualizedVolatility(FactorData):
    def __init__(self, price_histories_df, factor_data_df, days=20, annualization_factor=252):
        self.annualization_factor = annualization_factor
        super().__init__(self.compute(price_histories_df, days), f'annualzed_volatility_{days}_day')

    def compute(self, price_histories_df, days):
        return (FactorReturns(price_histories_df, days).factor_data.rolling(days).std() * (
                self.annualization_factor ** .5)).dropna()


class AverageDollarVolume(FactorData):
    def __init__(self, price_histories_df, days=20):
        super().__init__(self.compute(price_histories_df, days), f'average_dollar_volume_{days}_day')

    def compute(self, price_histories_df, days):
        return DailyDollarVolume(price_histories_df).factor_data.fillna(0).rolling(days).mean()


class MarketDispersion(FactorData):
    def __init__(self, price_histories_df, days=20):
        super().__init__(self.compute(price_histories_df, days), f'market_dispersion_{days}_day')

    def compute(self, price_histories_df, days=20):
        daily_returns = FactorReturns(price_histories_df, 1).factor_data.dropna()
        _market_dispersion = np.sqrt(
            np.nanmean(daily_returns.sub(daily_returns.mean(axis=1), axis=0) ** 2, axis=1)).reshape(-1, 1)
        for column in daily_returns.columns:
            daily_returns[column] = _market_dispersion
        return daily_returns.rolling(days).mean()


class MarketVolatility(FactorData):
    def __init__(self, price_histories_df, days=20, annualization_factor=252):
        self.annualization_factor = annualization_factor
        super().__init__(self.compute(price_histories_df, days), f'market_volatility_{days}_day')

    def compute(self, price_histories_df, days=20):
        daily_returns = FactorReturns(price_histories_df, 1).factor_data
        market_returns = daily_returns.mean(axis=1)
        market_returns_mu = market_returns.mean(axis=0)
        _market_volatility = np.sqrt(
            self.annualization_factor * market_returns.sub(market_returns_mu, axis=0) ** 2).values.reshape(-1, 1)
        for column in daily_returns.columns:
            daily_returns[column] = _market_volatility
        return daily_returns.rolling(days).mean()


# Date Parts
class FactorDateParts():
    def __init__(self, factors_df):
        self.factors_df = factors_df
        self.start_date = np.min(factors_df.index.get_level_values(0))
        self.end_date = np.max(factors_df.index.get_level_values(0))
        self.is_january()
        self.is_december()
        self.set_weekday_quarter_year()
        self.set_start_end_dates()

    def is_january(self):
        self.factors_df['is_January'] = (self.factors_df.index.get_level_values(0).month == 1).astype(int)

    def is_december(self):
        self.factors_df['is_December'] = (self.factors_df.index.get_level_values(0).month == 12).astype(int)

    def set_weekday_quarter_year(self):
        self.factors_df['weekday'] = self.factors_df.index.get_level_values(0).weekday
        self.factors_df['quarter'] = self.factors_df.index.get_level_values(0).quarter
        self.factors_df['year'] = self.factors_df.index.get_level_values(0).year

    def set_start_end_dates(self):
        # first day of month (Business Month Start)
        self.factors_df['month_start'] = (self.factors_df.index.get_level_values(0).isin(
            pd.date_range(start=self.start_date, end=self.end_date, freq='BMS'))).astype(int)

        # last day of month (Business Month)
        self.factors_df['month_end'] = (self.factors_df.index.get_level_values(0).isin(
            pd.date_range(start=self.start_date, end=self.end_date, freq='BM'))).astype(int)

        # first day of quarter (Business Month)
        self.factors_df['quarter_start'] = (self.factors_df.index.get_level_values(0).isin(
            pd.date_range(start=self.start_date, end=self.end_date, freq='BQS'))).astype(int)

        # last day of quarter (Business Month)
        self.factors_df['quarter_end'] = (self.factors_df.index.get_level_values(0).isin(
            pd.date_range(start=self.start_date, end=self.end_date, freq='BQ'))).astype(int)


# Factor Targets
class FactorReturnQuantiles(FactorData):
    def __init__(self, price_histories_df, quantiles=5, days=1):
        super().__init__(self.compute(price_histories_df, days, quantiles), f'logret_{days}_day_{quantiles}_quantiles')

    def compute(self, price_histories_df, days, quantiles):
        returns = FactorReturns(price_histories_df, days).factor_data
        return returns.apply(lambda x: pd.qcut(x, quantiles, labels=False, duplicates='drop'), axis=1)


# Utils

def get_sector_helper(stocks_df, sector_column, tickers):
    sector_data = stocks_df[sector_column][tickers]
    sector_helper = {}
    for sector in set(sector_data.values):
        sector_tickers = sector_data[lambda x: x == sector].index.to_list()
        sector_helper[sector] = sector_tickers
    return sector_helper


def filter_price_histories(price_histories, mask):
    return price_histories.iloc[:, price_histories.columns.get_level_values('Symbols').isin(mask)]


def compute_ai_alpha_score(samples, classifier):
    factor_probas = classifier.predict_proba(samples)
    classification_count = factor_probas.shape[1]

    prob_array = [*range(-int(classification_count / 2), 0)] + \
                 ([] if classification_count % 2 == 0 else [0]) + \
                 [*range(1, int(classification_count / 2) + 1)]

    return factor_probas.dot(prob_array)


def add_alpha_score(factor_data, classifier, ai_factor_name='AI_ALPHA'):
    alpha_score = compute_ai_alpha_score(factor_data, classifier)
    factor_data[ai_factor_name] = alpha_score
    return factor_data


def evaluate_alpha(data, pricing):
    clean_factor_data, unixt_factor_data = prepare_alpha_lense_factor_data(data.copy(), pricing)
    print('\n-----------------------\n')

    # Calculate Factor Returns and Sharpe Ratio
    factor_returns = get_factor_returns(clean_factor_data)
    factors_sharpe_ratio = compute_sharpe_ratio(factor_returns)

    # Show Results
    print('             Sharpe Ratios')
    print(factors_sharpe_ratio.round(2))
    plot_factor_returns(factor_returns)
    plot_factor_rank_autocorrelation(clean_factor_data)
    plot_basis_points_per_day_quantile(unixt_factor_data)
    return factor_returns, clean_factor_data, unixt_factor_data


def prepare_alpha_lense_factor_data(all_factors, pricing):
    clean_factor_data = {
        factor: al.utils.get_clean_factor_and_forward_returns(
            factor=factor_data, prices=pricing, periods=[1]) for factor, factor_data in all_factors.iteritems()}

    unixt_factor_data = {
        factor: factor_data.set_index(pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in factor_data.index.values],
            names=['date', 'asset'])) for factor, factor_data in clean_factor_data.items()}

    return clean_factor_data, unixt_factor_data


def eval_factor_and_add(factors_list, factor, pricing, min_sharpe_ratio=0.5):
    logger = logging.getLogger('trading_factors/eval_factor_and_add')
    logger.info(f'FACTOR_EVAL|{factor.factor_name}|{min_sharpe_ratio}...')
    factor_data = factor.for_al()
    clean_factor_data, unixt_factor_data = prepare_alpha_lense_factor_data(factor_data.to_frame().copy(),
                                                                           pricing)
    factor_returns = get_factor_returns(clean_factor_data)
    sharpe_ratio = compute_sharpe_ratio(factor_returns)['Sharpe Ratio'].values[0]

    if sharpe_ratio < min_sharpe_ratio:
        logger.info(f'FACTOR_EVAL|{factor.factor_name}|{min_sharpe_ratio}|{sharpe_ratio}|REJECTED')
        return
    logger.info(f'FACTOR_EVAL|{factor.factor_name}|{min_sharpe_ratio}|{sharpe_ratio}|ACCEPTED')
    factors_list.append(factor_data)


def get_factor_returns(factor_data):
    ls_factor_returns = pd.DataFrame()

    for factor, factor_data in factor_data.items():
        ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[:, 0]

    return ls_factor_returns


# Annualized Sharpe Ratios (daily = daily to annual, ...)
def compute_sharpe_ratio(df, frequency="daily"):
    if frequency == "daily":
        annualization_factor = np.sqrt(252)
    elif frequency == "monthly":
        annualization_factor = np.sqrt(12)
    else:
        # TODO: no conversion
        annualization_factor = 1

    # TODO: calculate the sharpe ratio and store it in a dataframe.
    # name the column 'Sharpe Ratio'.  
    # round the numbers to 2 decimal places
    df_sharpe = pd.DataFrame(data=annualization_factor * df.mean() / df.std(), columns=['Sharpe Ratio']).round(2)

    return df_sharpe


#########################################
# Plots
#########################################


def plot_factor_returns(factor_returns):
    (1 + factor_returns).cumprod().plot(title='Factor Returns')


def plot_factor_rank_autocorrelation(unixt_factor_data):
    ls_FRA = pd.DataFrame()

    for factor, factor_data in unixt_factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)

    ls_FRA.plot(title="Factor Rank Autocorrelation", ylim=(0.8, 1.0))


def plot_basis_points_per_day_quantile(unixt_factor_data):
    qr_factor_returns = pd.DataFrame()

    for factor, factor_data in unixt_factor_data.items():
        qr_factor_returns[factor] = al.performance.mean_return_by_quantile(factor_data)[0].iloc[:, 0]

    (10000 * qr_factor_returns).plot.bar(
        title='Quantile Analysis',
        subplots=True,
        sharey=True,
        layout=(4, 2),
        figsize=(14, 14),
        legend=False)


#########################################
# Risk Factors (Beta)
#########################################


def fit_pca(returns, num_factor_exposures, svd_solver):
    pca = PCA(num_factor_exposures, svd_solver=svd_solver)
    pca.fit(returns)
    return pca


def factor_betas(pca, factor_beta_indices, factor_beta_columns):
    return pd.DataFrame(pca.components_.T, index=factor_beta_indices, columns=factor_beta_columns)


def factor_returns(pca, returns, factor_return_indices, factor_return_columns):
    return pd.DataFrame(pca.transform(returns), index=factor_return_indices, columns=factor_return_columns)


def idiosyncratic_var_vector(returns, idiosyncratic_var_matrix):
    return pd.DataFrame(data=np.diag(idiosyncratic_var_matrix.values), index=returns.columns)


def factor_cov_matrix(factor_returns, ann_factor):
    return np.diag(factor_returns.var(axis=0, ddof=1) * ann_factor)


def portfolio_variance_using_factors(X, B, F, S):
    var_portfolio = X.T.dot(B.dot(F).dot(B.T) + S).dot(X)
    return np.sqrt(var_portfolio.sum())


def idiosyncratic_var_matrix(returns, factor_returns, factor_betas, ann_factor):
    common_returns_ = pd.DataFrame(np.dot(factor_returns, factor_betas.T), returns.index, returns.columns)
    residuals_ = (returns - common_returns_)
    return pd.DataFrame(np.diag(np.var(residuals_)) * ann_factor, returns.columns, returns.columns)


class RiskModelPCA(object):
    def __init__(self, returns, ann_factor, num_factor_exposures):
        # Configure
        self.factor_names_ = list(range(0, num_factor_exposures))
        self.tickers_ = returns.columns.values
        self.num_factor_exposures_ = num_factor_exposures
        self.ann_factor_ = ann_factor

        # Compute
        self.pca_ = fit_pca(returns, self.num_factor_exposures_, 'full')
        self.factor_betas_ = factor_betas(self.pca_, returns.columns.values, self.factor_names_)
        self.factor_returns_ = factor_returns(self.pca_, returns, returns.index, self.factor_names_)
        self.factor_cov_matrix_ = factor_cov_matrix(self.factor_returns_, self.ann_factor_)
        self.idiosyncratic_var_matrix_ = idiosyncratic_var_matrix(returns, self.factor_returns_,
                                                                  self.factor_betas_, self.ann_factor_)
        self.idiosyncratic_var_vector_ = idiosyncratic_var_vector(returns, self.idiosyncratic_var_matrix_)

    def compute_portfolio_variance(self, weights):
        X = np.array(weights).reshape((-1, 1))
        return portfolio_variance_using_factors(X, self.factor_betas_.values, self.factor_cov_matrix_,
                                                self.idiosyncratic_var_matrix_)
