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
from trading_functions import Data

class FactorData:
    def __init__(self, factor_data_df, factor_name='Alpha Factor'):
        
        self.factor_data = factor_data_df
        self.factor_name = factor_name
        
    def compute(self):
        pass
        
    def demean(self):
        return FactorData(self.factor_data.sub(self.factor_data.mean(axis=1), axis=0), self.factor_name)
            
    def rank(self):
        return FactorData(self.factor_data.rank(axis=1), self.factor_name)
    
    def zscore(self):
        return FactorData(self.factor_data.apply(stats.zscore, axis='columns'), self.factor_name)
    
    def smoothed(self, days=20):
        return FactorData(self.factor_data.rolling(window=days).mean(), self.factor_name + '_smoothed')
    
    def for_al(self, factor_name=None):
        if factor_name is not None:
            self.factor_name = factor_name
        alpha_lens_series = self.factor_data.stack()
        alpha_lens_series.name= self.factor_name
        return alpha_lens_series
        
class FactorReturns(FactorData):

    def __init__(self, price_histories_df, days=1):
        
        self.compute(price_histories_df, days)

    def compute(self, price_histories_df, days=1):
        self.factor_name = f'logret_{days}_day'
        self.factor_data = Returns().compute_log_returns(Data().get_close_values(price_histories_df), days)
        return self
    
class FactorMomentum(FactorData):
    
    def __init__(self, price_histories_df, days=252):
        
        self.compute(price_histories_df, days)
    
    def compute(self, price_histories_df, days=252):
        self.factor_name = f'momentum_{days}_day_logret'
        self.factor_data = FactorReturns(price_histories_df, days).factor_data
        return self
    
class FactorMeanReversion(FactorData):
    def __init__(self, price_histories_df, days=5):
        
        self.compute(price_histories_df, days)
        
    def compute(self, price_histories_df, days=5):
        # Note, The idea is that we are looking for underperormers to revert back towards the mean of the sector.
        #       Since the ranking will sort from under perormers to over performers, we reverse the factor value by 
        #       multiplying by -1, so that the largest underperormers are ranked higher.
        self.factor_name = f'mean_reversion_{days}_day_logret'
        self.factor_data = -FactorMomentum(price_histories_df, days).factor_data
        return self
    
class OvernightSentiment(FactorData):
    def __init__(self, price_histories_df, days=5):
        
        self.compute(price_histories_df, days)
    
    def compute(self, price_histories_df, days=5):
        self.factor_name = f'overnight_sentiment_{days}_day'
        close_prices = Data().get_close_values(price_histories_df)
        open_prices = Data().get_open_values(price_histories_df)
        self.factor_data = ((open_prices.shift(-1) - close_prices)  / close_prices).rolling(window=days, min_periods=1).sum()
        return self
    
# Universal Quant Features
class AnnualizedVolatility(FactorData):
    def __init__(self, price_histories_df, days=20, annualization_factor=252):
        self.annualization_factor = annualization_factor
        
        self.compute(price_histories_df, days)
    
    def compute(self, price_histories_df, days=20):
        self.factor_name = f'annualzed_volatility_{days}_day'
        self.factor_data = (FactorReturns(price_histories_df, days).factor_data.rolling(days).std() * (self.annualization_factor ** .5)).dropna()
        return self
    
class AverageDollarVolume(FactorData):
    def __init__(self, price_histories_df, days=5):
        
        self.compute(price_histories_df, days)
        
    def compute(self, price_histories_df, days=20):
        self.factor_name = f'average_dollar_volume_{days}_day'
        self.factor_data = (Data().get_values_by_date(price_histories_df, 'close') *
                            Data().get_values_by_date(price_histories_df, 'volume')).fillna(0).rolling(days).mean()
        return self
    
class MarketDispersion(FactorData):
    def __init__(self, price_histories_df, days=20, return_days=1):
        
        self.return_days = return_days
        self.compute(price_histories_df, days)
        
    def compute(self, price_histories_df, days=1):
        self.factor_name = f'market_dispersion{days}_day'
        returns = FactorReturns(price_histories_df, self.return_days).factor_data
        self.factor_data = np.sqrt(returns.sub(returns.mean(axis=1), axis=0)** 2).rolling(days).mean()
        return self

def af_demean(dataframe):
    return dataframe.sub(dataframe.mean(axis=1), axis=0)

def af_rank(dataframe):
    return dataframe.rank(axis=1)

def af_zscore(dataframe):
    return dataframe.apply(stats.zscore, axis='columns')

# need to add a demean flag, default True
def finalize_factor_data(raw_data_df, factor_name, rank_direction=1, demean=True):
    if demean:
        data_df = af_demean(raw_data_df)
    else:
        data_df = raw_data_df
        
    demeaned_ranked_zscored_df = (af_zscore(af_rank(data_df)) * rank_direction).stack()
    demeaned_ranked_zscored_df.name=factor_name
    return demeaned_ranked_zscored_df

def prepare_alpha_lense_factor_data(all_factors, pricing):
    clean_factor_data = {
    factor: al.utils.get_clean_factor_and_forward_returns(
        factor=factor_data, prices=pricing, periods=[1]) for factor, factor_data in all_factors.iteritems()}

    unixt_factor_data = {
        factor: factor_data.set_index(pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in factor_data.index.values],
            names=['date', 'asset'])) for factor, factor_data in clean_factor_data.items()}

    return clean_factor_data, unixt_factor_data


    
def sharpe_ratio(df, frequency="daily"):

    if frequency == "daily":
        # TODO: daily to annual conversion
        annualization_factor = np.sqrt(252)
    elif frequency == "monthly":
        #TODO: monthly to annual conversion
        annualization_factor = np.sqrt(12) 
    else:
        # TODO: no conversion
        annualization_factor = 1

    #TODO: calculate the sharpe ratio and store it in a dataframe.
    # name the column 'Sharpe Ratio'.  
    # round the numbers to 2 decimal places
    df_sharpe = pd.DataFrame(data=annualization_factor*df.mean()/df.std(), columns=['Sharpe Ratio']).round(2)

    return df_sharpe
    
def momentum(portfolio_price_histories, days=252):
    factor_name = f'momentum_{days}_day_factor_returns'
    raw_factor_data = Returns().compute_log_returns(Data().get_close_values(portfolio_price_histories), days)
    return finalize_factor_data(raw_factor_data, factor_name)

def mean_reversion_factor_returns(portfolio_price_histories, days=5):
    # Note, The idea is that we are looking for underperormers to revert back towards the mean of the sector.
    #       Since the ranking will sort from under perormers to over performers, we reverse the factor value by 
    #       multiplying by -1, so that the largest underperormers are ranked higher.
    factor_name = f'mean_reversion_{days}_day_factor_returns'
    raw_factor_data = Returns().compute_log_returns(Data().get_close_values(portfolio_price_histories), days)
    return finalize_factor_data(raw_factor_data, factor_name, -1)

def mean_reversion_factor_returns_smoothed(portfolio_price_histories, days=5):
    factor_name = f'mean_reversion_{days}_day_factor_returns_smoothed'
    raw_factor_data = mean_reversion_factor_returns(portfolio_price_histories, days).unstack(1).rolling(window=days).mean()
    return finalize_factor_data(raw_factor_data, factor_name, demean=False)

def mean_factor_returns_smoothed(portfolio_price_histories, days=5):
    factor_name = f'mean_{days}_day_factor_returns_smoothed'
    returns = Returns().compute_log_returns(Data().get_close_values(portfolio_price_histories), 1)
    raw_factor_data = returns.rolling(window=days).mean()
    return finalize_factor_data(raw_factor_data, factor_name, demean=False)

def overnight_sentiment(portfolio_price_histories, days=5):
    factor_name = f'overnight_sentiment_{days}_day'

    close_prices = Data().get_close_values(portfolio_price_histories)
    open_prices = Data().get_open_values(portfolio_price_histories)
    overnight_returns = ((open_prices.shift(-1) - close_prices)  / close_prices)
    raw_factor_data = overnight_returns.rolling(window=days, min_periods=1).sum()
    return finalize_factor_data(raw_factor_data, factor_name)

def overnight_sentiment_smoothed(portfolio_price_histories, days=5):
    factor_name = f'overnight_sentiment_{days}_day_smoothed'
    raw_factor_data = overnight_sentiment(portfolio_price_histories, days).unstack(1).rolling(window=days).mean()
    return finalize_factor_data(raw_factor_data, factor_name, demean=False)

# Universal Quant Features

def annualized_volatility(portfolio_price_histories, days=20, annualization_factor=252):
    factor_name = f'annualzed_volatility_{days}_day'
    raw_factor_data = Returns().compute_log_returns(Data().get_close_values(portfolio_price_histories), 1)
    raw_factor_data = (raw_factor_data.rolling(days).std() * (annualization_factor ** .5)).dropna()
    return finalize_factor_data(raw_factor_data, factor_name, demean=False)

def average_dollar_volume(portfolio_price_histories, days=20):
    factor_name = f'average_dollar_volume_{days}_day'
    close = Data
    raw_factor_data = Returns().compute_log_returns(Data().get_close_values(portfolio_price_histories), 1)
    raw_factor_data = (raw_factor_data.rolling(days).std() * (annualization_factor ** .5)).dropna()
    return finalize_factor_data(raw_factor_data, factor_name, demean=False)

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
