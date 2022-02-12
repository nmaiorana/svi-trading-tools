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

import json 

from pathlib import Path

ok_reason = ''
unauthorized_reason = 'Unauthorized'

date_format = '%Y-%m-%d'

## Authentication Data
# These items here are used to obtain an authorization token from TD Ameritrade. It involves navigating to web pages, so using a browser emulator # to navigate the page and set fields and submit pages.


class AmeritradeRest:
    
    def __init__(self, env_user_name='ameritradeuser', env_password='ameritradepw', env_client_id='ameritradeclientid'):
        
        self.configure_ameritrade(env_user_name, env_password, env_client_id)
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
    """
    Authentication Data
    These items here are used to obtain an authorization token from TD Ameritrade. It involves navigating to web pages, so using a browser emulator # to navigate 
    the page and set fields and submit pages.
    """
    def configure_ameritrade(self, env_user_name='ameritradeuser', env_password='ameritradepw', env_client_id='ameritradeclientid'):
        """
        In order to keep developers from setting usernames and passwords in a file, the credentials will be stored in environment varialbes. 
        The default values for the variable names are:
        - ameritradeuser    : Username
        - ameritradepw      : Password
        - ameritradeclientid: Client ID provided by Ameritrade Developer
        
        - These evnironment variable names can be overridden when the AmeritradeRest class is instantiated.
        """
        self.username = os.getenv(env_user_name)
        self.password = os.getenv(env_password)
        self.client_id = os.getenv(env_client_id)
        self.consumer_key = self.client_id + '@AMER.OAUTHAP'
    
    def authenticate(self):
        
        """
        Use the configured username, password and client id to obtain an authentication tocken from Ameritrade. This uses Ameritrade Developer's
        serverless authentication (https://developer.tdameritrade.com/content/simple-auth-local-apps). 
        
        Depending on how you have your Ameritrade account setup (I use 2 factor authentication) you may need to manually establish some browser history by using 
        the '--user-data-dir' Chrome option. By default it will put the Chrome browser history data (so you can set the trust this device option)
        in the user's home directory under '\svi-trading\chrome_browser_history'. If you want to override this location, set the value prior to calling
        the authenticate() method:
            
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
                        'response_type':'code',
                        'redirect_uri':self.callback_url,
                        'client_id':self.consumer_key
            }

            # build url
            login_url = requests.Request(method, self.oauth_url, params=payload).prepare().url

            # go to URL
            driver.get(login_url)
    
            #fill out form
            driver.find_element_by_id('username0').send_keys(self.username)
            driver.find_element_by_id('password1').send_keys(self.password)
            
            # click Login button
            driver.find_element_by_id('accept').click()
            
            # If we don't see the authorization page, then 2-factor auth is turned on and this device is not trusted.
            # If 2-factor authentication is turned on and this device has not been trusted then we have to wait for the user to 
            # manually complete authorization 
            
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
            
            # At this point you get an error back since there is no localhost server. But the URL contains the authentication code
            new_url = urllib.parse.unquote(driver.current_url)

            # grab the URL and parse it for the auth code
            code = new_url.split('code=')[1]

            # Use the auth code to get an auth token
            #define the headers
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
            authreply = requests.post(self.oath_token_url, headers=headers, data=payload)

            # convert json to dict
            self.authorization = authreply.json()
            return self.authorization
        except selexcept.NoSuchElementException as error:
            print(f'Error: {error}')
        except selexcept.WebDriverException as error:
            print(f'Error: {error}')
        finally:
            driver.close()
            
    def get_access_token(self):
        if self.authorization is None:
            raise RuntimeError('Not Authenticated') from None
        else:
            return self.authorization['access_token']
    
    ###########################################################################################################
    # Account Level Functions
    ###########################################################################################################

    def mask_account(self, account_id):
        masked_account = self.account_mask + account_id[-4:]
        self.unmasked_accounts[masked_account] = account_id
        return masked_account

    def unmask_account(self, masked_account):
        return self.unmasked_accounts[masked_account]

    def get_accounts(self):
        self.account_data = None
        # define endpoint
        endpoint = 'https://api.tdameritrade.com/v1/accounts'
        headers = {'Authorization': f'Bearer {self.get_access_token()}'}

        # make a request
        content = requests.get(url=endpoint, headers=headers)
        if content.reason != ok_reason:
            if content.reason == unauthorized_reason:
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
        if content.reason != ok_reason:
            if content.reason == unauthorized_reason:
                print('Error: {}'.format(content.reason))
                return None

        # convert data to data dictionary
        self.positions_data =  content.json()
        return self.positions_data
    
    def parse_portfolios_list(self):
        if self.positions_data is None:
            self.get_positions()
            
        if self.positions_data is None:
            print ("No positons data.")
            return None
        
        portfolio_list = []
        total_portfolio = {}
        for account in self.positions_data:
            securitiesAccount = account['securitiesAccount']
            masked_account_id = self.mask_account(securitiesAccount['accountId'])
            for position in securitiesAccount['positions']:
                instrument_data = {}
                instrument_data['account'] = masked_account_id
                instrument_data.update(position)
                instrument_data.update(position['instrument'])
                instrument_data.pop('instrument', None)

                portfolio_list.append(instrument_data)

        return pd.DataFrame.from_dict(portfolio_list).fillna(0)
    
    ###########################################################################################################
    # Ticker Level Functions
    ###########################################################################################################
    def get_price_histories(self, tickers, end_date=None, num_periods=1, silent=False):
        price_histories_df = pd.DataFrame()
        ticker_count = 0
        for symbol in tqdm(tickers, desc='Tickers', unit='Price Histories', disable=silent):
            ticker_price_history = self.get_daily_price_history(symbol, end_date, num_periods=num_periods)
            if ticker_price_history is not None:
                price_histories_df = price_histories_df.append([ticker_price_history])
                ticker_count += 1
                if ticker_count % 30 == 0:
                    time.sleep(10)
                    
        price_histories_df.reset_index(drop=True, inplace=True)
        return price_histories_df.sort_values(by=['date'])
    
    def get_daily_price_history(self, symbol, end_date=None, num_periods=1):
        if end_date is None:
            end_date = datetime.today().strftime(date_format)
        # define endpoint
        endpoint = f'https://api.tdameritrade.com/v1/marketdata/{symbol}/pricehistory'

        payload = {
                    'apikey': self.client_id,
                    'periodType': 'year',
                    'period': str(num_periods),
                    'frequencyType': 'daily',
                    'endDate': str(int(datetime.strptime(end_date, date_format).timestamp()) * 1000),
                    'needExtendedHoursData':'true'
        }

        # make a request
        content = requests.get(url=endpoint, params=payload)
        if content.reason != ok_reason:
            if content.reason == unauthorized_reason:
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
        # "session": "'NORMAL' or 'AM' or 'PM' or 'SEAMLESS'",
        # "duration": "'DAY' or 'GOOD_TILL_CANCEL' or 'FILL_OR_KILL'",
        # "orderType": "'MARKET' or 'LIMIT' or 'STOP' or 'STOP_LIMIT' or 'TRAILING_STOP' or 'MARKET_ON_CLOSE' or 'EXERCISE' or 'TRAILING_STOP_LIMIT' or 'NET_DEBIT' or 'NET_CREDIT' or 'NET_ZERO'",
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
            print(f'Placing SELL order on {self.mask_account(account)} for {row.longQuantity} shares of {row.symbol}...')
            result = self.place_order(account, row.symbol, row.assetType, row.longQuantity, 'SELL', session=session, duration=duration, orderType=orderType)
            results[row.symbol] = result
            
        return results

    def get_quotes(self, tickers):
        endpoint = f'https://api.tdameritrade.com/v1/marketdata/quotes'

        payload = {
                    'apikey': self.client_id,
                    'symbol': ",".join(tickers)
        }
        content = requests.get(url=endpoint, params=payload)
        return pd.DataFrame.from_dict(content.json(), orient='index')
        
