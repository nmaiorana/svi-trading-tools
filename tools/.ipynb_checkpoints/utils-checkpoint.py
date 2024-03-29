import logging
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image
import numpy as np
import pandas as pd
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

from tqdm.notebook import tqdm

import ssl

plt.rcParams['figure.figsize'] = (20, 8)
ssl._create_default_https_context = ssl._create_unverified_context


#####################
# Portfolio Functions
#####################


def save_port_data(dataframe, file_name):
    dataframe.to_csv(file_name, index=False)


def read_port_data(file_name):
    return pd.read_csv(file_name)


def save_price_histories(dataframe, file_name):
    dataframe.to_csv(file_name, index=False)


def read_price_histories(file_name):
    return pd.read_csv(file_name, parse_dates=['date'], index_col=False)


def get_account_portfolio_data(portfolios_df, account):
    return portfolios_df.query('account == "{}"'.format(account))


def get_account_value(account_portfolio_df):
    return account_portfolio_df['marketValue'].sum()


def get_investments_by_type(account_portfolio_df, investment_type='EQUITY'):
    return account_portfolio_df.query(f'assetType == "{investment_type}"')


def get_investment_symbols(account_portfolio_df):
    return list(account_portfolio_df['symbol'].values)


def get_holdings(account_portfolio_df, universe):
    current_holdings = account_portfolio_df[['symbol', 'marketValue', 'longQuantity']].set_index('symbol').sort_index()
    non_portfolio_symbols = universe - set(current_holdings.index.values)
    non_portfolio_values = pd.DataFrame.from_dict({symbol: [0, 0] for symbol in non_portfolio_symbols}, orient='index')
    non_portfolio_values.index.name = 'symbol'
    non_portfolio_values.columns = ['marketValue', 'longQuantity']
    return current_holdings.append(non_portfolio_values).sort_index()


def get_portfolio_weights(holdings):
    return (holdings / np.sum(holdings)).rename(columns={'marketValue': 'weight'}).sort_index()


########################
# Stock Daily Price & Returns Functions
########################


def get_close_values(price_histories_df):
    open_values = get_values_by_date(price_histories_df, 'open')
    close_values = get_values_by_date(price_histories_df, 'close')
    close_values = close_values.fillna(open_values.ffill())
    close_values = close_values.fillna(open_values.bfill())
    return close_values


def get_open_values(price_histories_df):
    open_values = get_values_by_date(price_histories_df, 'open')
    close_values = get_values_by_date(price_histories_df, 'close')
    open_values = open_values.fillna(close_values.ffill())
    open_values = open_values.fillna(close_values.bfill())
    return open_values


def get_values_by_date(price_histories_df, values):
    return price_histories_df.reset_index().pivot(index='date', columns='ticker', values=values).tz_localize('UTC',
                                                                                                             level='date')
def compute_log_returns(prices, lookahead=1):
    return np.log(prices / prices.shift(lookahead))[lookahead:].fillna(0)


def compute_log_returns_days(buy_prices, sell_prices):
    return np.log(sell_prices / buy_prices).to_frame()


########################
# Bulk Order Instructions
########################


def bulk_sell_order_instructions(stocks_df):
    sell_instructions = []
    for row in stocks_df.itertuples():
        sell_instructions.append(
            {"instruction": "SELL", "quantity": row.longQuantity,
             "instrument": {"symbol": row.symbol, "assetType": row.assetType}}
        )

    return sell_instructions


########################
# General analysis tools
########################


def plot_results(tree_sizes, train_score, oob_score, valid_score, legend, title, xlabel):
    plt.plot(tree_sizes, train_score, 'xb-');
    plt.plot(tree_sizes, oob_score, 'xg-');
    plt.plot(tree_sizes, valid_score, 'xr-');
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel('Accuracy')
    plt.legend(legend)
    # plt.ylim(y_range[0], y_range[1]);

    plt.show()


def plot_tree_classifier(clf, feature_names=None):
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        rotate=True)

    return Image(graphviz.Source(dot_data).pipe(format='png'))


def rank_features_by_importance(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    max_feature_name_length = max([len(feature) for feature in feature_names])

    print('      Feature{space: <{padding}}      Importance'.format(padding=max_feature_name_length - 8, space=' '))

    ranked_features = []
    for x_train_i in range(len(importances)):
        ranked_features.append(feature_names[indices[x_train_i]])
        print('{number:>2}. {feature: <{padding}} ({importance})'.format(
            number=x_train_i + 1,
            padding=max_feature_name_length,
            feature=feature_names[indices[x_train_i]],
            importance=importances[indices[x_train_i]]))

    return ranked_features


def train_valid_test_split(all_x, all_y, train_size, valid_size, test_size):
    """
    Generate the train, validation, and test dataset.

    Parameters
    ----------
    all_x : DataFrame
        All the input samples
    all_y : Pandas Series
        All the target values
    train_size : float
        The proportion of the data used for the training dataset
    valid_size : float
        The proportion of the data used for the validation dataset
    test_size : float
        The proportion of the data used for the test dataset

    Returns
    -------
    x_train : DataFrame
        The train input samples
    x_valid : DataFrame
        The validation input samples
    x_test : DataFrame
        The test input samples
    y_train : Pandas Series
        The train target values
    y_valid : Pandas Series
        The validation target values
    y_test : Pandas Series
        The test target values
    """
    assert train_size >= 0 and train_size <= 1.0
    assert valid_size >= 0 and valid_size <= 1.0
    assert test_size >= 0 and test_size <= 1.0
    print(train_size + valid_size + test_size)
    assert train_size + valid_size + test_size == 1.0

    all_dates = all_x.index.levels[0]
    train_end = int(all_dates.shape[0] * (train_size)) - 1
    valid_end = train_end + int(all_dates.shape[0] * valid_size)

    X_train = all_x.loc[:all_dates[train_end], ]
    X_valid = all_x.loc[all_dates[train_end + 1]:all_dates[valid_end], ]
    X_test = all_x.loc[all_dates[valid_end + 1]:]

    y_train = all_y.loc[:all_dates[train_end], ]
    y_valid = all_y.loc[all_dates[train_end + 1]:all_dates[valid_end], ]
    y_test = all_y.loc[all_dates[valid_end + 1]:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def non_overlapping_samples(x, y, n_skip_samples, start_i=0):
    """
    Get the non overlapping samples.

    Parameters
    ----------
    x : DataFrame
        The input samples
    y : Pandas Series
        The target values
    n_skip_samples : int
        The number of samples to skip
    start_i : int
        The starting index to use for the data
    
    Returns
    -------
    non_overlapping_x : 2 dimensional Ndarray
        The non overlapping input samples
    non_overlapping_y : 1 dimensional Ndarray
        The non overlapping target values
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 1

    dates = sorted(list(set(x.index.get_level_values(0))))
    non_overlapping_dates = dates[start_i::n_skip_samples + 1]
    x_sub = x.loc[non_overlapping_dates]
    y_sub = y.loc[non_overlapping_dates]

    return x_sub, y_sub


########################
# Pull tock data from wikitables
########################


def get_snp500() -> pd.DataFrame:
    return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                        header=0,
                        attrs={'id': 'constituents'},
                        index_col='Symbol')[0]


def get_dow():
    return pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average',
                        header=0,
                        attrs={'id': 'constituents'},
                        index_col='Symbol')[0]


########################
# Finvis Stock Sentiment
########################


finwiz_url = 'https://finviz.com/quote.ashx?t='


def get_finvis_stock_sentiment(tickers):
    logger = logging.getLogger('utils/get_finvis_stock_sentiment')
    logger.debug('Gathering Finvis stock sentiments')

    news_tables = {}
    pause_every = 15
    ticker_count = 0
    for ticker in tqdm(tickers, desc='Tickers', unit='Finvis Postings'):
        ticker_count += 1
        if ticker_count % pause_every == 0:
            time.sleep(2)
        url = finwiz_url + ticker
        req = Request(url=url, headers={'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
        try:
            response = urlopen(req)
        except:
            continue
        # Read the contents of the file into 'html'
        html = BeautifulSoup(response, features="lxml")
        # Find 'news-table' in the Soup and load it into 'news_table'
        news_table = html.find(id='news-table')
        # Add the table to our dictionary
        news_tables[ticker] = news_table
        response.close()

    parsed_news = []

    # Iterate through the news
    for file_name, news_table in tqdm(news_tables.items(), desc='News Tables', unit='News Table Items'):
        # Iterate through all tr tags in 'news_table'
        for x in news_table.findAll('tr'):
            # read the text from each tr tag into text
            # get text from a only
            if x.a is None:
                continue
            text = x.a.get_text()
            # split text in the td tag into a list
            date_scrape = x.td.text.split()
            # if the length of 'date_scrape' is 1, load 'time' as the only element

            if len(date_scrape) == 1:
                news_time = date_scrape[0]

            # else load 'date' as the 1st element and 'time' as the second    
            else:
                new_date = date_scrape[0]
                news_time = date_scrape[1]
            # Extract the ticker from the file name, get the string up to the 1st '_'  
            ticker = file_name.split('_')[0]

            # Append ticker, date, time and headline as a list to the 'parsed_news' list
            parsed_news.append([ticker, new_date, news_time, text])

    vader = SentimentIntensityAnalyzer()

    # Set column names
    columns = ['ticker', 'date', 'time', 'headline']

    # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
    parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

    # Convert the date column from string to datetime
    parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date

    return parsed_and_scored_news


#  Reduce the stock universe by 1 std of the mean of sentiment for all stocks in the last 40 days
def reduce_universe_by_sentiment(stock_universe):
    logger = logging.getLogger('utils/reduce_universe_by_sentiment')
    logger.debug(f'Number of stocks in universe: {len(stock_universe)}')
    parsed_and_scored_news = get_finvis_stock_sentiment(stock_universe).sort_values(by='date')
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.groupby(['ticker', 'date']).mean().fillna(0)
    # Unstack the column ticker
    mean_scores = mean_scores.unstack()
    # Get the cross-section of compound in the 'columns' axis
    mean_scores = mean_scores.xs('compound', axis="columns").transpose().fillna(0)
    # Get cusmum score of each stock
    cum_scores = mean_scores[-40:].cumsum(axis=0)
    current_scores = cum_scores.iloc[-1]
    mean_score = current_scores.mean()
    stdv_score = current_scores.std()
    cutoff = mean_score - stdv_score

    logger.debug(
        f'Mean Sentiment: {mean_score} with a standared deviation of: {stdv_score} providing a cutoff of: {cutoff}')
    reduced_stock_universe = current_scores.where(current_scores > cutoff).dropna().index.to_list()
    logger.debug(f'New number of stocks in universe: {len(reduced_stock_universe)}')
    return reduced_stock_universe
