#!/usr/bin/env python
# coding: utf-8

# # Trading Functions
# 
# These functions were derived from an AI for Trading course provided by Udacity. The functions were originally part of quizzes / exercises given during the course.


import pandas as pd
import numpy as np
import cvxpy as cvx
from sklearn.decomposition import PCA
import time
from datetime import datetime


class Data:
    
    def save_port_data(self, dataframe, file_name):
        dataframe.to_csv(file_name, index=False)

    def read_port_data(self, file_name):
        return pd.read_csv(file_name)
     
    def save_price_histories(self, dataframe, file_name):
        dataframe.to_csv(file_name, index=False)

    def read_price_histories(self, file_name):
        return pd.read_csv(file_name, parse_dates=['date'], index_col=False)
    
    def get_instrument_symbols(self, portfolio_df):
            return portfolio_df.columns.values.sort_index()
        
    def get_fundamental_symbols(self, fundamentals_df):
        return sorted(fundamentals_df['ticker'].unique())

    def get_values_by_date(self, fundamentals_df, values):
        return fundamentals_df.reset_index().pivot(index='date', columns='ticker', values=values)
    
    def get_close_values(self, fundamentals_df):
        open_values = self.get_values_by_date(fundamentals_df, 'open')
        close_values = self.get_values_by_date(fundamentals_df, 'close')
        close_values = close_values.fillna(open_values.ffill())
        close_values = close_values.fillna(open_values.bfill())
        return close_values
    
    def get_account_portfolio_data(self, portfolios_df, account):
        return portfolios_df.query('account == "{}"'.format(account))

    def resample_prices(self, close_prices, freq='M'):
        """
        Resample close prices for each ticker at specified frequency.
        Parameters
        ----------
        close_prices : DataFrame
            Close prices for each ticker and date
        freq : str
            What frequency to sample at
            For valid freq choices, see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

        Returns
        -------
        prices_resampled : DataFrame
            Resampled prices for each ticker and date
        """

        return close_prices.resample(freq).last().dropna()
    
class Portfolio:
    def get_portfolio_weights(self, market_values_df):
        return (market_values_df / self.get_account_value(market_values_df)).rename(columns={'marketValue':'weight'}).sort_index()
        
    def get_market_values(self, account_portfolio_df):
        return account_portfolio_df[['symbol', 'marketValue']].set_index('symbol').sort_index()
    
    def get_investment_symbols(self, market_values_df):
        return list(market_values_df.index)
        
    def get_investments_by_type(self, account_portfolio_df, investment_type='EQUITY'):
        return account_portfolio_df.query('assetType == "{}"'.format(investment_type))

    def get_account_value(self, market_values_df):
        return market_values_df.sum(axis=1).sum() 
    
    def portfolio_log_returns(self, close, weights, lookahead=1):
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
        
        #covariance matrix of returns
        covariance_returns = Returns().get_covariance_returns(returns)
        
        assert len(covariance_returns.shape) == 2
        assert len(index_weights.shape) == 1
        assert covariance_returns.shape[0] == covariance_returns.shape[1]  == index_weights.shape[0]

        # number of stocks m is number of rows of returns, and also number of index weights
        m = len(index_weights)

        # x variables (to be found with optimization)
        x = cvx.Variable(m)

        #portfolio variance, in quadratic form
        portfolio_variance = cvx.quad_form(x, covariance_returns)

        # euclidean distance (L2 norm) between portfolio and index weights
        distance_to_index = cvx.norm(x - index_weights)

        #objective function
        objective = cvx.Minimize(portfolio_variance + scale*distance_to_index)

        #constraints
        constraints = [x >= 0, sum(x) == 1]

        #use cvxpy to solve the objective
        problem = cvx.Problem(objective, constraints)
        result = problem.solve()

        #retrieve the weights of the optimized portfolio
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
            new_weights = self.get_optimal_weights(chunk_returns, index_weights.iloc[last_day-1])
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

        #covariance matrix of returns
        cov = Returns().get_covariance_returns(returns)
        
        weights = index_weights.values.T[0]
        assert len(cov.shape) == 2
        assert len(weights.shape) == 1
        assert cov.shape[0] == cov.shape[1]  == weights.shape[0]
        
        # number of stocks m is number of rows of returns, and also number of index weights
        m = len(weights)
        
        # x variables (to be found with optimization)
        x = cvx.Variable(m)

        #portfolio variance, in quadratic form
        portfolio_variance = cvx.quad_form(x, cov)

        # euclidean distance (L2 norm) between portfolio and index weights
        distance_to_index = cvx.norm(x - weights)

        #objective function
        objective = cvx.Minimize(portfolio_variance + scale*distance_to_index)

        #constraints
        constraints = [x >= 0, sum(x) == 1]

        #use cvxpy to solve the objective
        problem = cvx.Problem(objective, constraints)
        result = problem.solve()

        #retrieve the weights of the optimized portfolio
        x_values = x.value

        return pd.DataFrame(x_values, columns=['weight'], index=index_weights.index)
    
class Returns:
    
    def compute_log_returns(self, prices, lookahead=1):
        """
        Compute log returns for each ticker.

        Parameters
        ----------
        prices : DataFrame with prices for each ticker and date
            
        lookahead : Number of days to look ahead to compute returns

        Returns
        -------
        log_returns : DataFrame with Log returns for each ticker and date

        """

        return np.log(prices / prices.shift(lookahead))[lookahead:].fillna(0)
    
    def shift_returns(returns, shift_n):
        """
        Generate shifted returns

        Parameters
        ----------
        returns : DataFrame
            Returns for each ticker and date
        shift_n : int
            Number of periods to move, can be positive or negative

        Returns
        -------
        shifted_returns : DataFrame
            Shifted returns for each ticker and date
        """

        return returns.shift(shift_n)

    def calculate_arithmetic_rate_of_return(self, close):
        """
        Compute returns for each ticker and date in close.

        Parameters
        ----------
        close : DataFrame
            Close prices for each ticker and date

        Returns
        -------
        arithmnetic_returns : Series
            arithmnetic_returns at the end of the period for each ticker

        """
        return ((((close/close.shift(1)) - 1).cumsum() / len(close)).iloc[len(close)-1])
    
    def get_covariance_returns(self, returns):
        """
        Calculate covariance matrices.

        Parameters
        ----------
        returns : DataFrame
            Returns for each ticker and date

        Returns
        -------
        returns_covariance  : 2 dimensional Ndarray
            The covariance of the returns
        """
        return np.cov(returns.fillna(0).T.values)

class RiskModelPCA(object):
    def __init__(self, returns, ann_factor, num_factor_exposures):
        self.factor_names_ = list(range(0, num_factor_exposures))
        pca = self.fit_pca(returns, num_factor_exposures, 'full')
        self.factor_betas_ = self.factor_betas(pca, returns.columns.values, self.factor_names_)
        self.factor_returns_ = self.factor_returns(pca, returns, returns.index, self.factor_names_)
        self.factor_cov_matrix_ = self.factor_cov_matrix(self.factor_returns_, ann_factor)
        self.idiosyncratic_var_matrix_ = self.idiosyncratic_var_matrix(returns, self.factor_returns_, self.factor_betas_, ann_factor)

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
    
    def factor_cov_matrix(self, factor_returns, ann_factor):
        return np.diag(factor_returns.var(axis=0, ddof=1)*ann_factor)
    
    def compute_portfolio_variance(self, weights):
        X = np.array(weights).reshape((-1,1))
        return self.portfolio_variance_using_factors(X,self.factor_betas_.values,self.factor_cov_matrix_,self.idiosyncratic_var_matrix_)
    
    def portfolio_variance_using_factors(self, X, B, F, S):
        var_portfolio = X.T.dot(B.dot(F).dot(B.T) + S).dot(X)
        return var_portfolio
    
class Selection:
    
    def volatility(self, returns):
        return pd.DataFrame(returns.std(), columns=['volatility'], index=returns.columns)
    
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

        #TODO: Implement function

        dollar_volume = close * volume

        return dollar_volume.div(dollar_volume.sum(axis=1), axis=0)

    def get_top_n(self, prev_returns, top_n):
        """
        Select the top performing stocks

        Parameters
        ----------
        prev_returns : DataFrame
            Previous shifted returns for each ticker and date
        top_n : int
            The number of top performing stocks to get

        Returns
        -------
        top_stocks : DataFrame
            Top stocks for each ticker and date marked with a 1
        """
        top_stocks = pd.DataFrame(0, columns=prev_returns.columns, index=prev_returns.index)
        for index, row in prev_returns.iterrows():
            top_stocks.loc[index][row.nlargest(top_n).index.tolist()] = 1

        return top_stocks

def calculate_simple_moving_average(rolling_window, close):
    return close.rolling(window = rolling_window).mean()

def volatility(returns):
    return returns.std()

def annualized_volatility(close):
    monthly_close = resample_prices(close, freq='M')
    monthly_returns = compute_log_returns(monthly_close)
    monthly_volatility = volatility(monthly_returns)
    return np.sqrt(monthly_volatility * 12)

def estimate_volatility(prices, l):
    """Create an exponential moving average model of the volatility of a stock
    price, and return the most recent (last) volatility estimate.
    
    Parameters
    ----------
    prices : pandas.Series
        A series of adjusted closing prices for a stock.
        
    l : float
        The 'lambda' parameter of the exponential moving average model. Making
        this value smaller will cause the model to weight older terms less 
        relative to more recent terms.
        
    Returns
    -------
    last_vol : float
        The last element of your exponential moving averge volatility model series.
    
    """
    # TODO: Implement the exponential moving average volatility model and return the last value.
    alpha = 1.0 - l
    returns = np.log(prices / prices.shift(1)) **2
    volatility = np.sqrt(returns.ewm(alpha=alpha).mean())
    return volatility.values[-1]
    
def get_high_lows_lookback(high, low, lookback_days):
    """
    Get the highs and lows in a lookback window.
    
    Parameters
    ----------
    high : DataFrame
        High price for each ticker and date
    low : DataFrame
        Low price for each ticker and date
    lookback_days : int
        The number of days to look back
    
    Returns
    -------
    lookback_high : DataFrame
        Lookback high price for each ticker and date
    lookback_low : DataFrame
        Lookback low price for each ticker and date
    """
    lookback_high = high.shift(1).rolling(lookback_days).max()
    lookback_low =  low.shift(1).rolling(lookback_days).min()

    return lookback_high, lookback_low

def get_long_short(close, lookback_high, lookback_low):
    """
    Generate the signals long, short, and do nothing.
    
    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookback_high : DataFrame
        Lookback high price for each ticker and date
    lookback_low : DataFrame
        Lookback low price for each ticker and date
    
    Returns
    -------
    long_short : DataFrame
        The long, short, and do nothing signals for each ticker and date
    """
    long_short = ((close < lookback_low).astype(np.int64) * -1) + (close > lookback_high).astype(np.int64)
    
    return long_short

def clear_signals(signals, window_size):
    """
    Clear out signals in a Series of just long or short signals.
    
    Remove the number of signals down to 1 within the window size time period.
    
    Parameters
    ----------
    signals : Pandas Series
        The long, short, or do nothing signals
    window_size : int
        The number of days to have a single signal       
    
    Returns
    -------
    signals : Pandas Series
        Signals with the signals removed from the window size
    """
    # Start with buffer of window size
    # This handles the edge case of calculating past_signal in the beginning
    clean_signals = [0]*window_size
    
    for signal_i, current_signal in enumerate(signals):
        # Check if there was a signal in the past window_size of days
        has_past_signal = bool(sum(clean_signals[signal_i:signal_i+window_size]))
        # Use the current signal if there's no past signal, else 0/False
        clean_signals.append(not has_past_signal and current_signal)
        
    # Remove buffer
    clean_signals = clean_signals[window_size:]

    # Return the signals as a Series of Ints
    return pd.Series(np.array(clean_signals).astype(np.int), signals.index)


def filter_signals(signal, lookahead_days):
    """
    Filter out signals in a DataFrame.
    
    Parameters
    ----------
    signal : DataFrame
        The long, short, and do nothing signals for each ticker and date
    lookahead_days : int
        The number of days to look ahead
    
    Returns
    -------
    filtered_signal : DataFrame
        The filtered long, short, and do nothing signals for each ticker and date
    """
    
    return (signal == 1).astype(np.int64).apply(clear_signals, args=(lookahead_days,)) - (signal == -1).astype(np.int64).apply(clear_signals, args=(lookahead_days,))


def get_lookahead_prices(close, lookahead_days):
    """
    Get the lookahead prices for `lookahead_days` number of days.
    
    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookahead_days : int
        The number of days to look ahead
    
    Returns
    -------
    lookahead_prices : DataFrame
        The lookahead prices for each ticker and date
    """
    return close.shift(-lookahead_days)

def get_return_lookahead(close, lookahead_prices):
    """
    Calculate the log returns from the lookahead days to the signal day.
    
    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookahead_prices : DataFrame
        The lookahead prices for each ticker and date
    
    Returns
    -------
    lookahead_returns : DataFrame
        The lookahead log returns for each ticker and date
    """
    return np.log(lookahead_prices / close)

def get_signal_return(signal, lookahead_returns):
    """
    Compute the signal returns.
    
    Parameters
    ----------
    signal : DataFrame
        The long, short, and do nothing signals for each ticker and date
    lookahead_returns : DataFrame
        The lookahead log returns for each ticker and date
    
    Returns
    -------
    signal_return : DataFrame
        Signal returns for each ticker and date
    """
    return ((signal > 0).astype(np.int64) * lookahead_returns) + ((signal < 0).astype(np.int64) * lookahead_returns * -1)

from scipy.stats import kstest


def calculate_kstest(long_short_signal_returns):
    """
    Calculate the KS-Test against the signal returns with a long or short signal.
    
    Parameters
    ----------
    long_short_signal_returns : DataFrame
        The signal returns which have a signal.
        This DataFrame contains two columns, "ticker" and "signal_return"
    
    Returns
    -------
    ks_values : Pandas Series
        KS static for all the tickers
    p_values : Pandas Series
        P value for all the tickers
    """
    ks_values = {}
    p_values = {}
    
    return_values = long_short_signal_returns.signal_return
    normal_args = (np.mean(return_values), np.std(return_values, ddof=0))
    for index, ticker_return in long_short_signal_returns.groupby(['ticker']):
        ticker = ticker_return.ticker.values[0]
        signal_returns = ticker_return.signal_return.values
        t_stat, p_value = kstest(signal_returns, 'norm', normal_args)
        ks_values[ticker] = t_stat
        p_values[ticker] = p_value
    
    return pd.Series(ks_values), pd.Series(p_values)

def find_outliers(ks_values, p_values, ks_threshold, pvalue_threshold=0.05):
    """
    Find outlying symbols using KS values and P-values
    
    Parameters
    ----------
    ks_values : Pandas Series
        KS static for all the tickers
    p_values : Pandas Series
        P value for all the tickers
    ks_threshold : float
        The threshold for the KS statistic
    pvalue_threshold : float
        The threshold for the p-value
    
    Returns
    -------
    outliers : set of str
        Symbols that are outliers
    """
    outliers = (ks_values > ks_threshold) & (p_values < pvalue_threshold)
    
    return set(outliers.where(outliers).dropna().index.values)


