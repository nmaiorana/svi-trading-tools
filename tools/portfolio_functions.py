# # Trading Functions
# 
# These functions were derived from an AI for Trading course provided by Udacity. The functions were originally part
# of quizzes / exercises given during the course.


import pandas as pd
import numpy as np
import cvxpy as cvx
import trading_factors_yahoo as alpha_factors
from sklearn.decomposition import PCA
from scipy import stats
import alphalens as al
import time
from datetime import datetime
import os


class Portfolio:

    def portfolio_expected_returns(self, close, weights, lookahead=1):
        """
        Compute expected returns for the portfolio, assuming equal investment in each long/short stock.

        Parameters
        ----------
        close : DataFrame with Close price for each ticker and date

        lookahead : Number of days to look ahead to compute returns

        weights: Pandas Series with indexed by symbol containing each proportion of stock weight

        Returns
        -------
        portfolio_returns : DataFrame
            Expected portfolio returns for each ticker and date
        """
        return weights.values.T[0] * Returns().compute_log_returns(close, lookahead)

    def generate_dollar_volume_weights(self, close, volume):
        """
        Generate dollar volume weights.

        Parameters
        ----------
        close : DataFrame
            Close price for each ticker and date
        volume : str
            Volume for each ticker and date

        Returns
        -------
        dollar_volume_weights : DataFrame
            The dollar volume weights for each ticker and date
        """
        assert close.index.equals(volume.index)
        assert close.columns.equals(volume.columns)

        dollar_volume = close * volume

        return dollar_volume.div(dollar_volume.sum(axis=1), axis=0)

    def get_optimal_weights(self, returns, index_weights, scale=2.0):
        """
        Find the optimal weights.

        Parameters
        ----------
        returns : Pandas Dataframe
            2D array containing stock return series in each row.
        index_weights : Pandas Series
            Index weights for all tickers at a period in time
        scale : int
            The penalty factor for weights the deviate from the index 
        Returns
        -------
        x : 1 dimensional Ndarray
            The solution for x
        """

        # covariance matrix of returns
        covariance_returns = Returns().get_covariance_returns(returns)

        assert len(covariance_returns.shape) == 2
        assert len(index_weights.shape) == 1
        assert covariance_returns.shape[0] == covariance_returns.shape[1] == index_weights.shape[0]

        # number of stocks m is number of rows of returns, and also number of index weights
        m = len(index_weights)

        # x variables (to be found with optimization)
        x = cvx.Variable(m)

        # portfolio variance, in quadratic form
        portfolio_variance = cvx.quad_form(x, covariance_returns)

        # euclidean distance (L2 norm) between portfolio and index weights
        distance_to_index = cvx.norm(x - index_weights)

        # objective function
        objective = cvx.Minimize(portfolio_variance + scale * distance_to_index)

        # constraints
        constraints = [x >= 0, sum(x) == 1]

        # use cvxpy to solve the objective
        problem = cvx.Problem(objective, constraints)
        result = problem.solve()

        # retrieve the weights of the optimized portfolio
        x_values = x.value

        return pd.DataFrame(x_values, columns=['weight'], index=index_weights.index)

    def rebalance_portfolio(self, returns, index_weights, shift_size, chunk_size):
        """
        Get weights for each rebalancing of the portfolio.

        Parameters
        ----------
        returns : DataFrame
            Returns for each ticker and date
        index_weights : DataFrame
            Index weight for each ticker and date
        shift_size : int
            The number of days between each rebalance
        chunk_size : int
            The number of days to look in the past for rebalancing

        Returns
        -------
        all_rebalance_weights  : list of Ndarrays
            The ETF weights for each point they are rebalanced
        """

        assert returns.index.equals(index_weights.index)
        assert returns.columns.equals(index_weights.columns)
        assert shift_size > 0
        assert chunk_size >= 0

        rebalanced = list()
        for last_day in range(chunk_size, len(returns), shift_size):
            first_day = last_day - chunk_size
            chunk_returns = returns.iloc[first_day:last_day]
            new_weights = self.get_optimal_weights(chunk_returns, index_weights.iloc[last_day - 1])
            rebalanced.append(new_weights)

        return rebalanced

    def optimize_portfolio(self, returns, index_weights, scale=.00001):
        """
        Create a function that takes the return series of a set of stocks, the index weights,
        and scaling factor. The function will minimize a combination of the portfolio variance
        and the distance of its weights from the index weights.  
        The optimization will be constrained to be long only, and the weights should sum to one.

        Parameters
        ----------
        returns : Pandas Dataframe
            2D array containing stock return series in each row.

        index_weights : Pandas Series
            Indexed by symbol containing each proportion of stock weight

        scale : float
            The scaling factor applied to the distance between portfolio and index weights

        Returns
        -------
        x : np.ndarray
            A numpy ndarray containing the weights of the stocks in the optimized portfolio
        """

        # covariance matrix of returns
        cov = Returns().get_covariance_returns(returns)

        weights = index_weights.values.T[0]
        assert len(cov.shape) == 2
        assert len(weights.shape) == 1
        assert cov.shape[0] == cov.shape[1] == weights.shape[0]

        # number of stocks m is number of rows of returns, and also number of index weights
        m = len(weights)

        # x variables (to be found with optimization)
        x = cvx.Variable(m)

        # portfolio variance, in quadratic form
        portfolio_variance = cvx.quad_form(x, cov)

        # euclidean distance (L2 norm) between portfolio and index weights
        distance_to_index = cvx.norm(x - weights)

        # objective function
        objective = cvx.Minimize(portfolio_variance + scale * distance_to_index)

        # constraints
        constraints = [x >= 0, sum(x) == 1]

        # use cvxpy to solve the objective
        problem = cvx.Problem(objective, constraints)
        result = problem.solve()

        # retrieve the weights of the optimized portfolio
        x_values = x.value

        return pd.DataFrame(x_values, columns=['weight'], index=index_weights.index)
