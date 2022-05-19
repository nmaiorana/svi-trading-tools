

#!/usr/bin/env python
# coding: utf-8

# # Trading Functions
# 
# These functions were derived from an AI for Trading course provided by Udacity. The functions were originally part of quizzes / exercises given during the course.


import pandas as pd
import numpy as np
import cvxpy as cvx
from sklearn.decomposition import PCA
from scipy import stats
import alphalens as al
import time
from datetime import datetime
import os

from trading_functions import Returns

import utils

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
        return FactorData(pd.DataFrame(stats.zscore(self.factor_data, axis=1), index=self.factor_data.index, columns=self.factor_data.columns), self.factor_name)
    
    def smoothed(self, days=20):
        return FactorData(self.factor_data.rolling(window=days).mean(), self.factor_name + '_smoothed')
    
    def for_al(self, factor_name=None):
        if factor_name is not None:
            self.factor_name = factor_name
        alpha_lens_series = self.factor_data.stack()
        alpha_lens_series.name= self.factor_name
        return alpha_lens_series
    
class OpenPrices(FactorData):
    def __init__(self, price_histories_df, open_col='Open', close_col='Close'):
        self.open_col = open_col
        self.close_col = close_col
        self.compute(price_histories_df)
        
    def compute(self, price_histories_df):
        open_values = price_histories_df[self.open_col]
        close_values = price_histories_df[self.close_col]
        open_values = open_values.fillna(close_values.ffill())
        open_values = open_values.fillna(close_values.bfill())
        self.factor_name = 'open'
        self.factor_data = open_values
        return self
        
class ClosePrices(FactorData):
    def __init__(self, price_histories_df, open_col='Open', close_col='Close'):
        self.open_col = open_col
        self.close_col = close_col
        self.compute(price_histories_df)
        
    def compute(self, price_histories_df):
        open_values = price_histories_df[self.open_col]
        close_values = price_histories_df[self.close_col]
        close_values = close_values.fillna(open_values.ffill())
        close_values = close_values.fillna(open_values.bfill())
        self.factor_name = 'close'
        self.factor_data = close_values
        return self

class Volume(FactorData):
    def __init__(self, price_histories_df, volume_col='Volume'):
        self.volume_col = volume_col
        self.compute(price_histories_df)
        
    def compute(self, price_histories_df):
        volume_values = price_histories_df[self.volume_col]
        volume_values = volume_values.fillna(volume_values.ffill())
        self.factor_name = 'volume'
        self.factor_data = volume_values
        return self
        
class DailyDollarVolume(FactorData):
    def __init__(self, price_histories_df):
        self.compute(price_histories_df)

    def compute(self, price_histories_df):
        self.factor_name = f'daily_dollar_volume'
        self.factor_data = ClosePrices(price_histories_df).factor_data * Volume(price_histories_df).factor_data
        return self      
    
class FactorReturns(FactorData):
    def __init__(self, price_histories_df, days=1):
        self.compute(price_histories_df, days)

    def compute(self, price_histories_df, days):
        self.factor_name = f'returns_{days}_day'
        self.factor_data = ClosePrices(price_histories_df).factor_data.pct_change(days)
        return self
    
class FactorMomentum(FactorData):
    def __init__(self, price_histories_df, days=252):
        self.compute(price_histories_df, days)
    
    def compute(self, price_histories_df, days):
        self.factor_name = f'momentum_{days}_day'
        self.factor_data = FactorReturns(price_histories_df, days).factor_data
        return self
    
class FactorMeanReversion(FactorData):
    def __init__(self, price_histories_df, days=5):
        self.compute(price_histories_df, days)
        
    def compute(self, price_histories_df, days):
        # Note, The idea is that we are looking for underperormers to revert back towards the mean of the sector.
        #       Since the ranking will sort from under perormers to over performers, we reverse the factor value by 
        #       multiplying by -1, so that the largest underperormers are ranked higher.
        self.factor_name = f'mean_reversion_{days}_day_logret'
        self.factor_data = -FactorReturns(price_histories_df, days).factor_data
        return self
    
class CloseToOpen(FactorData):
    def __init__(self, price_histories_df):
        self.compute(price_histories_df)
    
    def compute(self, price_histories_df):
        self.factor_name = f'close_to_open'
        close_prices = ClosePrices(price_histories_df).factor_data
        open_prices = OpenPrices(price_histories_df).factor_data
        self.factor_data = ((open_prices.shift(-1) - close_prices)  / close_prices)
        return self
    
class TrailingOvernightReturns(FactorData):
    def __init__(self, price_histories_df, days=5):
        self.compute(price_histories_df, days)
    
    def compute(self, price_histories_df, days):
        self.factor_name = f'trailing_overnight_returns_{days}_day'
        self.factor_data = CloseToOpen(price_histories_df).factor_data.rolling(window=days, min_periods=1).sum()
        return self
    
# Universal Quant Features
class AnnualizedVolatility(FactorData):
    def __init__(self, price_histories_df, days=20, annualization_factor=252):
        self.annualization_factor = annualization_factor
        self.compute(price_histories_df, days)
    
    def compute(self, price_histories_df, days):
        self.factor_name = f'annualzed_volatility_{days}_day'
        self.factor_data = (FactorReturns(price_histories_df, days).factor_data.rolling(days).std() * (self.annualization_factor ** .5)).dropna()
        return self
    
class AverageDollarVolume(FactorData):
    def __init__(self, price_histories_df, days=20):
        self.compute(price_histories_df, days)
        
    def compute(self, price_histories_df, days):
        self.factor_name = f'average_dollar_volume_{days}_day'
        self.factor_data = DailyDollarVolume(price_histories_df).factor_data.fillna(0).rolling(days).mean()
        return self
    
class MarketDispersion(FactorData):
    def __init__(self, price_histories_df, days=20):
        self.compute(price_histories_df, days)
        
    def compute(self, price_histories_df, days=20):
        self.factor_name = f'market_dispersion_{days}_day'
        daily_returns = FactorReturns(price_histories_df, 1).factor_data.dropna()
        _market_dispersion = np.sqrt(np.nanmean(daily_returns.sub(daily_returns.mean(axis=1), axis=0)** 2, axis=1)).reshape(-1, 1)
        for column in daily_returns.columns:
            daily_returns[column] = _market_dispersion
        self.factor_data = daily_returns.rolling(days).mean()
        return self
    
class MarketVolatility(FactorData):
    def __init__(self, price_histories_df, days=20, annualization_factor=252):
        self.annualization_factor = annualization_factor
        self.compute(price_histories_df, days)
        
    def compute(self, price_histories_df, days=20):
        self.factor_name = f'market_volatility_{days}_day'
        daily_returns = FactorReturns(price_histories_df, 1).factor_data
        market_returns = daily_returns.mean(axis=1)
        market_returns_mu = market_returns.mean(axis=0)
        _market_volatility = np.sqrt(self.annualization_factor * market_returns.sub(market_returns_mu, axis=0)** 2).values.reshape(-1, 1)
        for column in daily_returns.columns:
            daily_returns[column] = _market_volatility
        self.factor_data = daily_returns.rolling(days).mean()
        return self
    
# Date Parts
class FactorDateParts():
    def __init__(self, factors_df):
        self.factors_df = factors_df
        self.start_date = np.min(factors_df.index.get_level_values(0))
        self.end_date = np.max(factors_df.index.get_level_values(0))
        self.isJanuary()
        self.isDecember()
        self.setWeekDayQuarterYear()
        self.setStartEndDates()
    
    def isJanuary(self):
        self.factors_df['is_January'] = (self.factors_df.index.get_level_values(0).month == 1).astype(int) 
        
    def isDecember(self):
        self.factors_df['is_December'] = (self.factors_df.index.get_level_values(0).month == 12).astype(int) 
    
    def setWeekDayQuarterYear(self):
        self.factors_df['weekday'] = self.factors_df.index.get_level_values(0).weekday
        self.factors_df['quarter'] = self.factors_df.index.get_level_values(0).quarter
        self.factors_df['year'] = self.factors_df.index.get_level_values(0).year
        
    def setStartEndDates(self):
        #first day of month (Business Month Start)
        self.factors_df['month_start'] = (self.factors_df.index.get_level_values(0).isin(pd.date_range(start=self.start_date, end=self.end_date, freq='BMS'))).astype(int)

        #last day of month (Business Month)
        self.factors_df['month_end'] = (self.factors_df.index.get_level_values(0).isin(pd.date_range(start=self.start_date, end=self.end_date, freq='BM'))).astype(int)

        #first day of quarter (Business Month)
        self.factors_df['quarter_start'] = (self.factors_df.index.get_level_values(0).isin(pd.date_range(start=self.start_date, end=self.end_date, freq='BQS'))).astype(int)

        #last day of quarter (Business Month)
        self.factors_df['quarter_end'] = (self.factors_df.index.get_level_values(0).isin(pd.date_range(start=self.start_date, end=self.end_date, freq='BQ'))).astype(int)
        
#Factor Targets
class FactorReturnQuantiles(FactorData):
    def __init__(self, price_histories_df, quantiles=5, return_days=1):
        self.return_days = return_days
        self.compute(price_histories_df, quantiles)
    
    def compute(self, price_histories_df, quantiles=5):
        self.factor_name = f'logret_{self.return_days}_day_{quantiles}_quantiles'
        returns = FactorReturns(price_histories_df, self.return_days).factor_data
        self.factor_data = returns.apply(lambda x: pd.qcut(x, quantiles, labels=False,  duplicates='drop'), axis=1)
        return self

# Utils

def get_sector_helper(stocks_df, sector_column, tickers):
    sector_data = stocks_df[sector_column][tickers]
    sector_helper = {}
    for sector in set(sector_data.values):
        sector_tickers = sector_data[lambda x: x==sector].index.to_list()
        sector_helper[sector] = sector_tickers
    return sector_helper

def filter_price_histories(price_histories, mask):
    return price_histories.iloc[:, price_histories.columns.get_level_values('Symbols').isin(mask)]

def compute_ai_alpha_score(samples, classifier):
    factor_probas = classifier.predict_proba(samples)
    classification_count = factor_probas.shape[1]
    
    prob_array = [*range(-int(classification_count/2), 0)] + \
        ([] if classification_count%2 == 0 else [0]) + \
        [*range(1, int(classification_count/2) + 1)]

    
    return factor_probas.dot(prob_array)

def add_alpha_score(factor_data, classifier, ai_factor_name='AI_ALPHA'):
    alpha_score = compute_ai_alpha_score(factor_data, classifier)
    factor_data[ai_factor_name] = alpha_score
    return factor_data

def evaluate_ai_alpha(data, pricing):
    clean_factor_data, unixt_factor_data = prepare_alpha_lense_factor_data(data.copy(), pricing)
    print('\n-----------------------\n')
    
    # Calculate Factor Returns and Sharpe Ratio
    factor_returns = get_factor_returns(clean_factor_data)
    factors_sharpe_ratio = sharpe_ratio(factor_returns)
    
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

# Not sure if this is used any longer
'''
def build_factor_data(factor_data, pricing):
    return {factor_name: al.utils.get_clean_factor_and_forward_returns(factor=data, prices=pricing, periods=[1])
        for factor_name, data in factor_data.iteritems()}
'''

def plot_factor_returns(factor_returns):
    (1+factor_returns).cumprod().plot(title='Factor Returns')

def plot_factor_rank_autocorrelation(unixt_factor_data):
    ls_FRA = pd.DataFrame()

    for factor, factor_data in unixt_factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)

    ls_FRA.plot(title="Factor Rank Autocorrelation", ylim=(0.8, 1.0))

def plot_basis_points_per_day_quantile(unixt_factor_data):
    qr_factor_returns = pd.DataFrame()

    for factor, factor_data in unixt_factor_data.items():
        qr_factor_returns[factor] = al.performance.mean_return_by_quantile(factor_data)[0].iloc[:, 0]

    (10000*qr_factor_returns).plot.bar(
        title='Quantile Analysis',
        subplots=True,
        sharey=True,
        layout=(4,2),
        figsize=(14, 14),
        legend=False)

def get_factor_returns(factor_data):
    ls_factor_returns = pd.DataFrame()

    for factor, factor_data in factor_data.items():
        ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[:, 0]

    return ls_factor_returns
    
# Annulized Sharpe Ratios (daily = daily to annual, ...)
def sharpe_ratio(df, frequency="daily"):

    if frequency == "daily":
        annualization_factor = np.sqrt(252)
    elif frequency == "monthly":
        annualization_factor = np.sqrt(12) 
    else:
        # TODO: no conversion
        annualization_factor = 1

    #TODO: calculate the sharpe ratio and store it in a dataframe.
    # name the column 'Sharpe Ratio'.  
    # round the numbers to 2 decimal places
    df_sharpe = pd.DataFrame(data=annualization_factor*df.mean()/df.std(), columns=['Sharpe Ratio']).round(2)

    return df_sharpe

class RiskModelPCA(object):
    def __init__(self, returns, ann_factor, num_factor_exposures):
        # Configure
        self.factor_names_ = list(range(0, num_factor_exposures))
        self.tickers_ = returns.columns.values
        self.num_factor_exposures_ = num_factor_exposures
        self.ann_factor_ = ann_factor
        
        # Compute
        self.pca_ = self.fit_pca(returns, self.num_factor_exposures_, 'full')
        self.factor_betas_ = self.factor_betas(self.pca_, returns.columns.values, self.factor_names_)
        self.factor_returns_ = self.factor_returns(self.pca_, returns, returns.index, self.factor_names_)
        self.factor_cov_matrix_ = self.factor_cov_matrix(self.factor_returns_, self.ann_factor_)
        self.idiosyncratic_var_matrix_ = self.idiosyncratic_var_matrix(returns, self.factor_returns_, self.factor_betas_, self.ann_factor_)
        self.idiosyncratic_var_vector_ = self.idiosyncratic_var_vector(returns, self.idiosyncratic_var_matrix_)

    def fit_pca(self, returns, num_factor_exposures, svd_solver):
        pca = PCA(num_factor_exposures, svd_solver=svd_solver)
        pca.fit(returns)
        return  pca
    
    def factor_betas(self, pca, factor_beta_indices, factor_beta_columns):
        return pd.DataFrame(pca.components_.T, index=factor_beta_indices, columns=factor_beta_columns)

    def factor_returns(self, pca, returns, factor_return_indices, factor_return_columns):
        return pd.DataFrame(pca.transform(returns), index=factor_return_indices, columns=factor_return_columns)

    def idiosyncratic_var_matrix(self, returns, factor_returns, factor_betas, ann_factor):
        common_returns_ = pd.DataFrame(np.dot(factor_returns, factor_betas.T), returns.index, returns.columns)
        residuals_ = (returns - common_returns_)
        return pd.DataFrame(np.diag(np.var(residuals_))*ann_factor, returns.columns, returns.columns)
    
    def idiosyncratic_var_vector(self, returns, idiosyncratic_var_matrix):
        return pd.DataFrame(data=np.diag(idiosyncratic_var_matrix.values), index=returns.columns)
    
    def factor_cov_matrix(self, factor_returns, ann_factor):
        return np.diag(factor_returns.var(axis=0, ddof=1)*ann_factor)
    
    def compute_portfolio_variance(self, weights):
        X = np.array(weights).reshape((-1,1))
        return self.portfolio_variance_using_factors(X,self.factor_betas_.values,self.factor_cov_matrix_,self.idiosyncratic_var_matrix_)
    
    def portfolio_variance_using_factors(self, X, B, F, S):
        var_portfolio = X.T.dot(B.dot(F).dot(B.T) + S).dot(X)
        return np.sqrt(var_portfolio.sum())
