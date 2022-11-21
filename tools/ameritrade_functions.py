import urllib
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import selenium.common.exceptions as selexcept
from webdriver_manager.chrome import ChromeDriverManager
import time
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import os
import configparser
import json 

from pathlib import Path

DEFAULT_CONFIG_LOCATION = '~/td_config.ini'
CONFIG_SECTION = 'TD_CONFIG'
ENV_CLIENT_ID_VARIABLE = 'env_client_id_variable'
ENV_PW_VARIABLE = 'env_pw_variable'
ENV_USER_VARIABLE = 'env_user_variable'
AMER_OAUTH_APP = '@AMER.OAUTHAP'
REFRESH_AUTH_TIME = 'refresh_auth_time'
PRIMARY_AUTH_TIME = 'primary_auth_time'
OK_REASON = ''
NOT_AUTHORIZED_REASON = 'Unauthorized'
DATE_FORMAT = '%Y-%m-%d'

# # Authentication Data These items here are used to obtain an authorization token from TD Ameritrade. It involves
# navigating to web pages, so using a browser emulator # to navigate the page and set fields and submit pages.


class AmeritradeRest:
    
    def __init__(self, config=None, config_file=DEFAULT_CONFIG_LOCATION):
        # TODO: Add ability to pass optional config path name

        self.username = None
        self.password = None
        self.client_id = None
        self.configure_ameritrade(config, config_file)
        # This is used to cache credentials in Chromedriver. You will have to manually log in the first time.
        self.user_data_dir = str(Path.home()) + r'\svi-trading\chrome_browser_history'

        self.callback_url = r'http://localhost'
        self.oauth_url = r'https://auth.tdameritrade.com/auth'
        self.oath_token_url = r'https://api.tdameritrade.com/v1/oauth2/token'
        
        self.unmasked_accounts = {}

        self.authorization = None
        self.account_data = None
        self.positions_data = None
        
        self.account_mask = '#---'

    ###########################################################################################################
    # Authentication Functions
    ###########################################################################################################
    """Authentication Data These items here are used to obtain an authorization token from TD Ameritrade. It involves 
    navigating to web pages, so using a browser emulator # to navigate the page and set fields and submit pages. """
    def configure_ameritrade(self, config=None, config_file=DEFAULT_CONFIG_LOCATION):
        """
        In order to keep developers from setting usernames and passwords in a file, the credentials will be stored in
        environment varialbes. The default values for the variable names are:
        - ameritradeuser    : Username
        - ameritradepw      : Password
        - ameritradeclientid: Client ID provided by Ameritrade Developer
        - These environment variable names can be overridden when the AmeritradeRest class is instantiated.

        TODO: Add ability to pass config file path and get data from there for the env names
        """
        if config is None:
            config = configparser.ConfigParser()
            config.read(os.path.expanduser(config_file))
            config = config[CONFIG_SECTION]

        if config is not None:
            self.username = os.getenv(config[ENV_USER_VARIABLE])
            self.password = os.getenv(config[ENV_PW_VARIABLE])
            self.client_id = os.getenv(config[ENV_CLIENT_ID_VARIABLE])

    def get_consumer_key(self):
        if self.client_id is None:
            return None

        return self.client_id + AMER_OAUTH_APP
    
    def authenticate(self):

        """

        Use the configured username, password and client id to obtain an authentication tocken from Ameritrade. This
        uses Ameritrade Developer's serverless authentication
        (https://developer.tdameritrade.com/content/simple-auth-local-apps).

        Depending on how you have your Ameritrade account setup (I use 2-factor authentication) you may need to
        manually establish some browser history by using the '--user-data-dir' Chrome option. By default, it will put
        the Chrome browser history data (so you can set the trust this device option) in the user's home directory
        under /svi-trading/chrome_browser_history.

        If you want to override this location, set the value prior to
        calling the authenticate() method:

            import ameritrade_functions as amc

            td_ameritrade = amc.AmeritradeRest()
            td_ameritrade.user_data_dir = 'somedirectory'
            td_ameritrade.authenticate()

        The code will attempt to allow the manual entry of the 2-factor data and should identify the authorization screen to move forward with obtaining
        the authentication token.
        """

        chrome_options = Options()
        chrome_options.add_argument('--user-data-dir='+self.user_data_dir)
        chrome_options.headless = False
        driver = webdriver.Chrome(ChromeDriverManager(log_level=0).install(), options=chrome_options)
        driver.minimize_window()  
        try:
            # define the components of the url
            method = 'GET'
            payload = {
                        'response_type': 'code',
                        'redirect_uri': self.callback_url,
                        'client_id': self.get_consumer_key()
            }

            # build url
            login_url = requests.Request(method, self.oauth_url, params=payload).prepare().url

            # go to URL
            driver.get(login_url)
    
            # fill out form
            driver.find_element_by_id('username0').send_keys(self.username)
            driver.find_element_by_id('password1').send_keys(self.password)
            
            # click Login button
            driver.find_element_by_id('accept').click()
            
            # If we don't see the authorization page, then 2-factor auth is turned on and this device is not trusted.
            # If 2-factor authentication is turned on and this device has not been trusted then we have to wait for 
            # the user to manually complete authorization 
            
            authorization_page = None
            while authorization_page is None:
                try:
                    authorization_page = driver.find_element_by_id('stepup_authorization0')
                except selexcept.NoSuchElementException as error:
                    driver.switch_to.window(driver.current_window_handle)
                    driver.maximize_window()  
                    time.sleep(2)

            # click allow on authorization screen
            driver.find_element_by_id('accept').click()

            # give it a second
            time.sleep(1)
            
            # At this point you get an error back since there is no localhost server. But the URL contains the
            # authentication code
            new_url = urllib.parse.unquote(driver.current_url)

            # grab the URL and parse it for the auth code
            code = new_url.split('code=')[1]

            # Use the auth code to get an auth token
            # define the headers
            headers = {'Content-Type':'application/x-www-form-urlencoded'}

            # define payload
            payload = {
                'grant_type':'authorization_code',
                'access_type': 'offline',
                'code': code,
                'client_id': self.client_id,
                'redirect_uri': 'http://localhost'
            }

            # post the data to get a token
            auth_reply = requests.post(self.oath_token_url, headers=headers, data=payload)

            # convert json to dict
            self.authorization = auth_reply.json()
            authorization_time = datetime.now()
            self.authorization[PRIMARY_AUTH_TIME] = authorization_time
            self.authorization[REFRESH_AUTH_TIME] = authorization_time
            return self.authorization
        except selexcept.NoSuchElementException as error:
            print(f'Error: {error}')
        except selexcept.WebDriverException as error:
            print(f'Error: {error}')
        finally:
            driver.close()

    def get_authorization(self):
        if self.authorization is None:
            # if no env get from file
            # if refresh has expired, get new
            pass

        return self.authorization

    def get_access_token(self):
        if self.get_authorization() is None:
            raise RuntimeError('Not Authenticated') from None
        else:
            return self.get_authorization()['access_token']

    def get_primary_auth_time(self):
        return self.get_authorization()[PRIMARY_AUTH_TIME]

    def get_refresh_auth_time(self):
        return self.get_authorization()[REFRESH_AUTH_TIME]

    ###########################################################################################################
    # Account Level Functions
    ###########################################################################################################

    def mask_account(self, account_id):
        masked_account = self.account_mask + account_id[-4:]
        self.unmasked_accounts[masked_account] = account_id
        return masked_account

    def unmask_account(self, masked_account):
        return self.unmasked_accounts[masked_account]

    def refresh_data(self):
        self.get_accounts()
        self.get_positions()

    def get_accounts(self):
        self.account_data = None
        # define endpoint
        endpoint = 'https://api.tdameritrade.com/v1/accounts'
        headers = {'Authorization': f'Bearer {self.get_access_token()}'}

        # make a request
        content = requests.get(url=endpoint, headers=headers)
        if content.reason != OK_REASON:
            if content.reason == NOT_AUTHORIZED_REASON:
                print(f'Error: {content.reason}')
                return None

        # convert data to data dictionary
        self.account_data = content.json()
        return self.parse_accounts()
        
    def parse_accounts(self):
        if self.account_data is None:
            self.get_accounts()
            
        if self.account_data is None:
            print ("No account data.")
            return None
            
        accounts_dict = {}
        for securitiesAccount in self.account_data:
            account = {}
            account_details = securitiesAccount['securitiesAccount']
            account['accountId'] = self.mask_account(account_details['accountId'])
            account['initialBalances_cashBalance'] =  account_details['initialBalances']['cashBalance']
            account['initialBalances_totalCash'] =  account_details['initialBalances']['totalCash']
            account['initialBalances_equity'] =  account_details['initialBalances']['equity']
            account['initialBalances_moneyMarketFund'] =  account_details['initialBalances']['moneyMarketFund']
            account['currentBalances_cashBalance'] =  account_details['currentBalances']['cashBalance']
            account['currentBalances_equity'] =  account_details['currentBalances']['equity']
            account['currentBalances_moneyMarketFund'] =  account_details['currentBalances']['moneyMarketFund']
            accounts_dict[account['accountId']] = account

        accounts_df = pd.DataFrame.from_dict(accounts_dict, orient='index')
        accounts_df['current_return'] = np.log(accounts_df['currentBalances_equity'] / accounts_df['initialBalances_equity'] )
        accounts_df.set_index('accountId', drop=True, inplace=True)
        return accounts_df

    def get_positions(self):
        # define endpoint
        endpoint = 'https://api.tdameritrade.com/v1/accounts'
        headers = {'Authorization': 'Bearer {}'.format(self.authorization['access_token'])}
        payload = {'fields': 'positions'}

        # make a request
        content = requests.get(url=endpoint, headers=headers, params=payload)
        if content.reason != OK_REASON:
            if content.reason == NOT_AUTHORIZED_REASON:
                print('Error: {}'.format(content.reason))
                return None

        # convert data to data dictionary
        self.positions_data = content.json()
        return self.positions_data
    
    def parse_portfolios_list(self):
        if self.positions_data is None:
            self.get_positions()
            
        if self.positions_data is None:
            print ("No positions data.")
            return None
        
        portfolio_list = []
        total_portfolio = {}
        for account in self.positions_data:
            securities_account = account['securitiesAccount']
            masked_account_id = self.mask_account(securities_account['accountId'])
            for position in securities_account['positions']:
                instrument_data = {'account': masked_account_id}
                instrument_data.update(position)
                instrument_data.update(position['instrument'])
                instrument_data.pop('instrument', None)

                portfolio_list.append(instrument_data)

        return pd.DataFrame.from_dict(portfolio_list).fillna(0).set_index(['account', 'symbol'])

    def get_account_portfolio_data(self, masked_account, investment_type=None) -> pd.DataFrame:
        full_portfolio = self.parse_portfolios_list()
        if investment_type is None:
            return full_portfolio.query(f'account == "{masked_account}"')
        else:
            return full_portfolio.query(f'account == "{masked_account}" and assetType == "{investment_type}"')

    def get_investment_symbols(self, masked_account, investment_type=None):
        return self.get_account_portfolio_data(masked_account, investment_type)\
            .index.get_level_values('symbol').tolist()

    def get_market_values(self, masked_account, investment_type=None) -> pd.Series:
        if investment_type is None:
            return self.get_account_portfolio_data(masked_account)['marketValue']
        else:
            return self.get_account_portfolio_data(masked_account, investment_type)['marketValue']

    def get_account_value(self, masked_account, investment_type=None) -> float:
        if investment_type is None:
            return self.get_market_values(masked_account).sum()
        else:
            return self.get_market_values(masked_account, investment_type).sum()

    def get_holdings(self, masked_account, investment_type=None, symbols=None) -> pd.Series:
        account_portfolio = self.get_account_portfolio_data(masked_account, investment_type)
        if symbols is None:
            symbols = account_portfolio.index.get_level_values('symbol').tolist()

        symbols = set(symbols)
        account_portfolio = account_portfolio[account_portfolio.index.get_level_values('symbol').isin(symbols)]
        current_holdings = account_portfolio[['marketValue', 'longQuantity']]
        non_portfolio_symbols = symbols - set(account_portfolio.index.get_level_values('symbol').values)
        if len(non_portfolio_symbols) == 0:
            return current_holdings

        non_portfolio_values = pd.DataFrame.from_dict(
            {(masked_account, symbol): [0, 0] for symbol in non_portfolio_symbols}, orient='index'
        )
        non_portfolio_values.index.name = 'symbol'
        non_portfolio_values.columns = ['marketValue', 'longQuantity']
        return current_holdings.append(non_portfolio_values).sort_index()

    def get_portfolio_weights(self, masked_account, investment_type=None, symbols=None) -> pd.Series:
        holdings = self.get_holdings(masked_account, investment_type, symbols)['marketValue']
        return (holdings / np.sum(holdings)).sort_index()
    ###########################################################################################################
    # Ticker Level Functions
    ###########################################################################################################

    def get_price_histories(self, tickers, end_date=None, num_periods=1, silent=False):
        price_histories_df = pd.DataFrame()
        ticker_count = 0
        for symbol in tqdm(tickers, desc='Tickers', unit='Price Histories', disable=silent):
            ticker_price_history = self.get_daily_price_history(symbol, end_date, num_periods=num_periods)
            if ticker_price_history is not None:
                price_histories_df = pd.concat([price_histories_df, ticker_price_history])
                ticker_count += 1
                if ticker_count % 30 == 0:
                    time.sleep(10)
                    
        price_histories_df.reset_index(drop=True, inplace=True)
        return price_histories_df.sort_values(by=['date'])
    
    def get_daily_price_history(self, symbol, end_date=None, num_periods=1):
        if end_date is None:
            end_date = datetime.today().strftime(DATE_FORMAT)
        # define endpoint
        endpoint = f'https://api.tdameritrade.com/v1/marketdata/{symbol}/pricehistory'

        payload = {
                    'apikey': self.client_id,
                    'periodType': 'year',
                    'period': str(num_periods),
                    'frequencyType': 'daily',
                    'endDate': str(int(datetime.strptime(end_date, DATE_FORMAT).timestamp()) * 1000),
                    'needExtendedHoursData': 'true'
        }

        # make a request
        content = requests.get(url=endpoint, params=payload)
        if content.reason != OK_REASON:
            if content.reason == NOT_AUTHORIZED_REASON:
                print(f'Error: {content.reason}')
                return None

        # convert data to data dictionary
        price_history = content.json()
        if 'candles' not in price_history:
            print(f'No candle data for {symbol}')
            return None
        
        candles = price_history['candles']
        if len(candles) == 0:
            print(f'Empty candle data for {symbol}')
            return None
        
        price_history_df = pd.DataFrame(price_history['candles'])

        price_history_df['ticker'] = price_history['symbol']
        price_history_df['date'] = pd.to_datetime(price_history_df['datetime'], unit='ms').dt.normalize()
        price_history_df.drop(['datetime'], inplace=True, axis=1)
        return price_history_df

    def get_fundamental(self, tickers):
        endpoint = f'https://api.tdameritrade.com/v1/instruments'

        payload = {
                    'apikey': self.client_id,
                    'symbol': ",".join(tickers),
                    'projection': 'fundamental'
        }
        content = requests.get(url=endpoint, params=payload)
        
        fundamental_data = content.json()
        fundamental_list = []
        for ticker in fundamental_data:
            ticker_fundamentals = {}
            ticker_fundamentals.update(fundamental_data[ticker])
            ticker_fundamentals.update(fundamental_data[ticker]['fundamental'])
            ticker_fundamentals.pop('fundamental', None)

            fundamental_list.append(ticker_fundamentals)

        return pd.DataFrame.from_dict(fundamental_list).fillna(0)
    
    def place_order(self, account, symbol, assetType='EQUITY', quantity=0, instruction='SELL', session='NORMAL', duration='DAY', orderType='MARKET'):
        # "session": "'NORMAL' or 'AM' or 'PM' or 'SEAMLESS'", "duration": "'DAY' or 'GOOD_TILL_CANCEL' or
        # 'FILL_OR_KILL'", "orderType": "'MARKET' or 'LIMIT' or 'STOP' or 'STOP_LIMIT' or 'TRAILING_STOP' or
        # 'MARKET_ON_CLOSE' or 'EXERCISE' or 'TRAILING_STOP_LIMIT' or 'NET_DEBIT' or 'NET_CREDIT' or 'NET_ZERO'",
        endpoint = f'https://api.tdameritrade.com/v1/accounts/{account}/savedorders'
        headers = {
                    'Authorization': 'Bearer {}'.format(self.authorization['access_token']),
                    'Content-type':'application/json'
                  }

        payload = {
                    'complexOrderStrategyType': 'NONE',
                    'session': session,
                    'duration': duration,
                    'orderType': orderType,
                    'orderStrategyType': 'SINGLE',
                    'orderLegCollection': [
                        {'instruction': instruction, 'quantity': quantity, 'instrument': {'symbol': symbol, 'assetType': assetType}}
                    ]
                }
        
        content = requests.post(url=endpoint, headers=headers, json=payload)
        if content.status_code != requests.codes.ok:
            print('Error: {}'.format(content.reason))
            return None        
        print(f'Placed {instruction} order on {self.mask_account(account)} for {quantity} shares of {symbol}')
        return content
    
    def place_bulk_sell_orders(self, account, stocks_df, session='NORMAL', duration='DAY', orderType='MARKET'):
        results = {}
        for row in stocks_df.itertuples():
            print(f'Placing SELL order on {self.mask_account(account)} for {row.longQuantity} shares of {row[0][1]}...')
            result = self.place_order(account, row[0][1], row.assetType, row.longQuantity, 'SELL', session=session, duration=duration, orderType=orderType)
            results[row[0][1]] = result
            
        return results

    def get_quotes(self, tickers):
        endpoint = f'https://api.tdameritrade.com/v1/marketdata/quotes'

        payload = {
                    'apikey': self.client_id,

                    'symbol': ",".join(tickers)
        }
        content = requests.get(url=endpoint, params=payload)
        return pd.DataFrame.from_dict(content.json(), orient='index')
