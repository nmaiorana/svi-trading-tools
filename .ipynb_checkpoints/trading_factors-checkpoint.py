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

def af_demean(dataframe):
    return dataframe.sub(dataframe.mean(axis=1), axis=0)

def af_rank(dataframe):
    return dataframe.rank(axis=1)

def af_zscore(dataframe):
    return dataframe.apply(stats.zscore, axis='columns')

def finalize_factor_data(raw_data_df, factor_name, rank_direction=1):
    demeaned_ranked_zscored_df = (af_zscore(af_rank(af_demean(raw_data_df))) * rank_direction).stack()
    demeaned_ranked_zscored_df.name=factor_name
    return demeaned_ranked_zscored_df

def momentum(portfolio_price_histories, days=252):
    factor_name = f'momentum_{days}_day_factor_returns'
    raw_factor_data = Returns().compute_log_returns(Data().get_close_values(portfolio_price_histories), days)
    return finalize_factor_data(raw_factor_data, factor_name)

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

def mean_revision_factor_returns(portfolio_price_histories, days=5):
    # Note, The idea is that we are looking for underperormers to revert back towards the mean of the sector.
    #       Since the ranking will sort from under perormers to over performers, we reverse the factor value by 
    #       multiplying by -1, so that the largest underperormers are ranked higher.
    factor_name = f'mean_revision_{days}_day_factor_returns'
    raw_factor_data = Returns().compute_log_returns(Data().get_close_values(portfolio_price_histories), days)
    return finalize_factor_data(raw_factor_data, factor_name, -1)

def mean_revision_factor_returns_smoothed(portfolio_price_histories, days=5):
    factor_name = f'mean_revision_{days}_day_factor_returns_smoothed'
    raw_factor_data = mean_revision_factor_returns(portfolio_price_histories, days).unstack(1).rolling(window=days).mean()
    return finalize_factor_data(raw_factor_data, factor_name)

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
    return finalize_factor_data(raw_factor_data, factor_name)

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
