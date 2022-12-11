import unittest
from unittest.mock import MagicMock
import tools.ameritrade_functions as amc
import os
import json
import configparser
from datetime import datetime, timedelta
from pathlib import Path

TEST_USER = 'TEST_USER'
TEST_PW = 'TEST_PW'
TEST_CLIENT_ID = 'TEST_CLIENT_ID'
TEST_CONFIG_PATH = 'test_config/td_config.ini'
TEST_MASKED_ACCOUNT = '#---9216'

TEST_AUTHORIZATION = {
    "access_token": "TEST_AUTH_TOKEN",
    "refresh_token": "TEST_REFRESH_TOKEN",
    "scope": "PlaceTrades AccountAccess MoveMoney",
    "expires_in": 1800,
    "refresh_token_expires_in": 7776000,
    "token_type": "Bearer",
    "primary_auth_time": datetime.now().isoformat(),
    "refresh_auth_time": datetime.now().isoformat()
}


def get_test_config():
    config = configparser.ConfigParser()
    config.read(TEST_CONFIG_PATH)
    return config[amc.CONFIG_SECTION]


def set_env_test_config():
    test_config = get_test_config()
    os.environ[test_config[amc.ENV_USER_VARIABLE]] = TEST_USER
    os.environ[test_config[amc.ENV_PW_VARIABLE]] = TEST_PW
    os.environ[test_config[amc.ENV_CLIENT_ID_VARIABLE]] = TEST_CLIENT_ID


# # Unit Tests

# ## Test Configuration
# 
# Configure the Ameritrade client to use the username, password and client id stored in environment variables.


class TestConfiguration(unittest.TestCase):
    test_config = get_test_config()

    @classmethod
    def setUpClass(cls):
        set_env_test_config()

    def test_config_object(self):
        class_under_test = amc.AmeritradeRest(config=self.test_config)
        self.assertEqual(TEST_USER, class_under_test.username)
        self.assertEqual(TEST_PW, class_under_test.password)
        self.assertEqual(TEST_CLIENT_ID, class_under_test.client_id)
        self.assertEqual(TEST_CLIENT_ID+amc.AMER_OAUTH_APP, class_under_test.get_consumer_key())

    def test_config_file(self):
        class_under_test = amc.AmeritradeRest(config_file=TEST_CONFIG_PATH)
        self.assertEqual(TEST_USER, class_under_test.username)
        self.assertEqual(TEST_PW, class_under_test.password)
        self.assertEqual(TEST_CLIENT_ID, class_under_test.client_id)
        self.assertEqual(TEST_CLIENT_ID+amc.AMER_OAUTH_APP, class_under_test.get_consumer_key())

    def test_get_consumer_key(self):
        class_under_test = amc.AmeritradeRest(config=self.test_config)
        orig_client_id = class_under_test.client_id
        class_under_test.client_id = None
        self.assertIsNone(class_under_test.get_consumer_key())
        class_under_test.client_id = orig_client_id
        self.assertIsNotNone(class_under_test.get_consumer_key())
        self.assertEqual(TEST_CLIENT_ID + amc.AMER_OAUTH_APP, class_under_test.get_consumer_key())


class TestAuthorizationTokens(unittest.TestCase):
    test_config = get_test_config()
    test_authorization = None

    @classmethod
    def setUpClass(cls) -> None:
        set_env_test_config()

    def setUp(self) -> None:
        self.test_authorization = {
            "access_token": "TEST_AUTH_TOKEN",
            "refresh_token": "TEST_REFRESH_TOKEN",
            "scope": "PlaceTrades AccountAccess MoveMoney",
            "expires_in": 1800,
            "refresh_token_expires_in": 7776000,
            "token_type": "Bearer",
            "primary_auth_time": datetime.now().isoformat(),
            "refresh_auth_time": datetime.now().isoformat()
        }

        self.class_under_test = amc.AmeritradeRest(config=self.test_config)
        self.class_under_test.authorization = self.test_authorization
        mock_authenticate = MagicMock(name='authenticate')
        self.class_under_test.authenticate = mock_authenticate

    def tearDown(self) -> None:
        self.remove_authorization_file()

    def remove_authorization_file(self):
        saved_file = Path(self.class_under_test.get_authorization_file_location())
        if saved_file.is_file():
            os.remove(self.class_under_test.get_authorization_file_location())

    def store_authorization(self):
        self.remove_authorization_file()
        self.class_under_test.authorization = self.test_authorization
        self.class_under_test.save_authorization()

    def test_get_authorization_file_location(self):
        self.assertIsNotNone(self.class_under_test.get_authorization_file_location())

    def test_save_authorization(self):
        self.remove_authorization_file()
        self.class_under_test.authorization = self.test_authorization
        self.class_under_test.save_authorization()
        saved_file = Path(self.class_under_test.get_authorization_file_location())
        self.assertTrue(saved_file.is_file())

    def test_load_authorization(self):
        self.store_authorization()
        self.class_under_test.load_authorization()
        self.assertIsNotNone(self.class_under_test.authorization)

    def test_is_access_token_expired(self):
        seconds_to_expire = self.test_authorization[amc.EXPIRES_IN]
        self.class_under_test.authorization[amc.PRIMARY_AUTH_TIME] = \
            (datetime.now() - timedelta(seconds=(seconds_to_expire + 100))).isoformat()
        self.assertTrue(self.class_under_test.is_access_token_expired())

    def test_is_access_token_not_expired(self):
        seconds_to_expire = self.test_authorization[amc.EXPIRES_IN]
        self.class_under_test.authorization[amc.PRIMARY_AUTH_TIME] = \
            (datetime.now()).isoformat()
        self.assertFalse(self.class_under_test.is_access_token_expired())

    def test_is_refresh_token_expired(self):
        seconds_to_expire = self.test_authorization[amc.REFRESH_TOKEN_EXPIRES_IN]
        self.class_under_test.authorization[amc.REFRESH_AUTH_TIME] = \
            (datetime.now() - timedelta(seconds=(seconds_to_expire + 100))).isoformat()
        self.assertTrue(self.class_under_test.is_refresh_token_expired())

    def test_is_refresh_token_not_expired(self):
        seconds_to_expire = self.test_authorization[amc.REFRESH_TOKEN_EXPIRES_IN]
        self.class_under_test.authorization[amc.REFRESH_AUTH_TIME] = \
            (datetime.now()).isoformat()
        self.assertFalse(self.class_under_test.is_refresh_token_expired())

    def test_get_expiry_time(self):
        expires_in = self.test_authorization[amc.EXPIRES_IN]
        refresh_token_expires_in = self.test_authorization[amc.REFRESH_TOKEN_EXPIRES_IN]
        self.assertEqual(expires_in, self.class_under_test.get_access_token_expiry_time())
        self.assertEqual(refresh_token_expires_in, self.class_under_test.get_refresh_token_expiry_time())

    def test_get_authorization_via_authenticate(self):
        # Testing:
        # - No Authorization, No Authorization File, call authenticate

        self.remove_authorization_file()
        self.class_under_test.authorization = None
        self.class_under_test.get_authorization()
        self.class_under_test.authenticate.assert_called()

    def test_get_authorization_via_file(self):
        # Testing:
        # - No Authorization, Load from File

        # with No authorization and no stored authorization file, expect the authenticate method to be called.

        self.store_authorization()
        self.class_under_test.authorization = None
        self.assertIsNotNone(self.class_under_test.get_authorization())

    def test_get_authorization_via_refresh(self):
        # Testing:
        # - Authorization token expired, call to refresh token
        # - Authorization token expired, refresh token expired, call authenticate

        # with No authorization and no stored authorization file, expect the authenticate method to be called.

        seconds_to_expire = self.test_authorization[amc.EXPIRES_IN]
        self.class_under_test.authorization[amc.PRIMARY_AUTH_TIME] = \
            (datetime.now() - timedelta(seconds=seconds_to_expire + 1)).isoformat()
        seconds_to_expire = self.test_authorization[amc.REFRESH_TOKEN_EXPIRES_IN]
        self.class_under_test.authorization[amc.REFRESH_AUTH_TIME] = \
            (datetime.now()).isoformat()

        mock_refresh_token = MagicMock(name='refresh_token')
        self.class_under_test.refresh_access_token = mock_refresh_token
        self.class_under_test.get_authorization()
        self.class_under_test.refresh_access_token.assert_called()

    def test_get_access_token(self):
        # Unauthenticated
        self.remove_authorization_file()
        self.class_under_test.authorization = None
        with self.assertRaises(RuntimeError) as cm:
            self.class_under_test.get_access_token()

# ## Account Level Functions


class TestAccountLevelFunctions(unittest.TestCase):
    def test_mask_account(self):
        class_under_test = amc.AmeritradeRest()
        self.assertEqual(class_under_test.mask_account('123456789'), '#---6789')
        class_under_test.account_mask = "*****"
        self.assertEqual(class_under_test.mask_account('12345678'), '*****5678')
        
    def test_unmasked_accounts(self):
        class_under_test = amc.AmeritradeRest()
        masked_account = class_under_test.mask_account('12345678')
        self.assertEqual(class_under_test.unmask_account(masked_account), '12345678')        


class TestAccountFunctions(unittest.TestCase):
    masked_account_1 = "#---1111"
    masked_account_2 = "#---2222"
    masked_account_3 = "#---3333"

    @classmethod
    def setUpClass(cls) -> None:
        cls.class_under_test = amc.AmeritradeRest()
        # Get test account data
        with open('test_data/account_data.json', 'r') as openfile:
            # Reading from json file
            cls.class_under_test.account_data = json.load(openfile)
        # Get test positions data
        with open('test_data/positions_data.json', 'r') as openfile:
            # Reading from json file
            cls.class_under_test.positions_data = json.load(openfile)

    def test_parse_accounts(self):
        accounts_list = self.class_under_test.parse_accounts()
        self.assertEqual(accounts_list.shape, (3, 8))
        self.assertIn('currentBalances_cashBalance', accounts_list.columns)
        self.assertIn('currentBalances_equity', accounts_list.columns)
        self.assertIn(self.masked_account_1, accounts_list.index)
        
    def test_parse_portfolios_list(self):
        portfolio_list = self.class_under_test.parse_portfolios_list()
        self.assertGreater(len(portfolio_list), 0)
        self.assertEqual(portfolio_list.columns.shape, (15,))
        self.assertIn('assetType', portfolio_list.columns)
        self.assertIn('cusip', portfolio_list.columns)
        self.assertIn('marketValue', portfolio_list.columns)
        self.assertIn('longQuantity', portfolio_list.columns)
        self.assertIn('type', portfolio_list.columns)
        self.assertIn(self.masked_account_1, portfolio_list.index.get_level_values('account'))
        self.assertIn('USB', portfolio_list.index.get_level_values('symbol'))

    def test_get_account_portfolio_data(self):
        account_portfolio = self.class_under_test.get_account_portfolio_data(self.masked_account_1)
        self.assertIn('USB', account_portfolio.index.get_level_values('symbol'))
        account_portfolio = self.class_under_test.get_account_portfolio_data(self.masked_account_1, 'EQUITY')
        self.assertIn('USB', account_portfolio.index.get_level_values('symbol'))
        account_portfolio = self.class_under_test.get_account_portfolio_data(self.masked_account_1, 'CASH_EQUIVALENT')
        self.assertIn('MMDA1', account_portfolio.index.get_level_values('symbol'))

    def test_get_market_values(self):
        market_values = self.class_under_test.get_market_values(self.masked_account_1)
        self.assertEqual(5, len(market_values))
        market_values = self.class_under_test.get_market_values(self.masked_account_1, 'EQUITY')
        self.assertEqual(4, len(market_values))

    def test_get_account_value(self):
        account_value = self.class_under_test.get_account_value(self.masked_account_1)
        self.assertAlmostEqual(6796.88, account_value, 2)
        account_value = self.class_under_test.get_account_value(self.masked_account_1, 'EQUITY')
        self.assertAlmostEqual(5760.43, account_value, 2)

    def test_get_holdings(self):
        holdings = self.class_under_test.get_holdings(self.masked_account_1)
        self.assertEqual(5, len(holdings))
        self.assertListEqual([1036.45, 1036.450], holdings.loc[self.masked_account_1, 'MMDA1'].to_list())
        holdings = self.class_under_test.get_holdings(self.masked_account_1, 'EQUITY')
        self.assertEqual(4, len(holdings))
        self.assertListEqual([2410.99, 11.085], holdings.loc[self.masked_account_1, 'UNP'].to_list())
        holdings = self.class_under_test.get_holdings(self.masked_account_1, 'INVALID_TYPE')
        self.assertEqual(0, len(holdings))
        holdings = self.class_under_test.get_holdings(self.masked_account_1, 'EQUITY', ['AAPL', 'GOOG', 'USB', 'UNP'])
        self.assertEqual(4, len(holdings))
        self.assertListEqual([0.0, 0.0], holdings.loc[self.masked_account_1, 'AAPL'].to_list())
        self.assertListEqual([2410.99, 11.085], holdings.loc[self.masked_account_1, 'UNP'].to_list())

    def test_get_investment_symbols(self):
        symbols = self.class_under_test.get_investment_symbols(self.masked_account_1)
        self.assertIn('USB', symbols)
        symbols = self.class_under_test.get_investment_symbols(self.masked_account_1, 'EQUITY')
        self.assertIn('USB', symbols)
        symbols = self.class_under_test.get_investment_symbols(self.masked_account_1, 'CASH_EQUIVALENT')
        self.assertIn('MMDA1', symbols)

    def test_get_portfolio_weights(self):
        weights = self.class_under_test.get_portfolio_weights(self.masked_account_1)
        self.assertAlmostEqual(0.34, weights[self.masked_account_1, 'USB'], 2)
        weights = self.class_under_test.get_portfolio_weights(self.masked_account_1, 'EQUITY')
        self.assertAlmostEqual(0.40, weights[self.masked_account_1, 'USB'], 2)
        weights = self.class_under_test.get_portfolio_weights(self.masked_account_1, 'CASH_EQUIVALENT')
        self.assertAlmostEqual(1.0, weights[self.masked_account_1, 'MMDA1'], 2)


# # Integration Tests
#
# Requires you to have a developer account with Ameritrade (https://developer.tdameritrade.com).
#
# These settings are configured via environment variables.

# ## Stock Information Functions

class TestStockInformationFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.class_under_test = amc.AmeritradeRest()

    def test_get_daily_price_history(self):
        price_histories = self.class_under_test.get_daily_price_history('AAPL', '2022-01-03')
        self.assertEqual(len(price_histories), 253)
        self.assertEqual(price_histories.date.min().strftime('%Y-%m-%d'), '2021-01-04')
        self.assertEqual(price_histories.date.max().strftime('%Y-%m-%d'), '2022-01-03')
        self.assertEqual(price_histories.ticker.unique()[0], 'AAPL')

    def test_get_price_histories(self):
        price_histories = self.class_under_test.get_price_histories(['AAPL', 'GOOG'], '2022-01-03', silent=True)
        self.assertEqual(506, len(price_histories))
        self.assertEqual(price_histories.date.min().strftime('%Y-%m-%d'), '2021-01-04')
        self.assertEqual(price_histories.date.max().strftime('%Y-%m-%d'), '2022-01-03')
        self.assertEqual(2, len(price_histories.ticker.unique()))

    def test_get_fundamental(self):
        fundamentals = self.class_under_test.get_fundamental(['AAPL', 'GOOG'])
        self.assertEqual(len(fundamentals), 2)
        self.assertEqual(len(fundamentals.symbol.unique()), 2)
        self.assertEqual(fundamentals[fundamentals.symbol == 'AAPL'].cusip.values[0], '037833100')
        self.assertEqual(fundamentals[fundamentals.symbol == 'GOOG'].cusip.values[0], '02079K107')

    def test_get_quotes(self):
        quotes = self.class_under_test.get_quotes(['AAPL', 'GOOG'])
        self.assertEqual(len(quotes), 2)
        self.assertEqual(len(quotes.symbol.unique()), 2)
        self.assertEqual(quotes[quotes.symbol == 'AAPL'].cusip.values[0], '037833100')
        self.assertEqual(quotes[quotes.symbol == 'GOOG'].cusip.values[0], '02079K107')


# ## Test Authenticated Functions

class TestAuthenticated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.class_under_test = amc.AmeritradeRest()
        cls.class_under_test.load_authorization()

    def test_get_authorization(self):
        self.assertIsNotNone(self.class_under_test.get_authorization())

    def test_authentication(self):
        self.class_under_test.authenticate()
        self.assertGreater(len(self.class_under_test.authorization), 0)
        self.assertIsNotNone(self.class_under_test.get_primary_auth_time())
        self.assertIsNotNone(self.class_under_test.get_refresh_auth_time())
        self.assertIsInstance(self.class_under_test.get_access_token_expiry_time(), int)
        self.assertIsInstance(self.class_under_test.get_refresh_token_expiry_time(), int)

    def test_get_access_token(self):
        self.assertIsNotNone(self.class_under_test.get_access_token())

    def test_refresh_access_token(self):
        current_auth_time = self.class_under_test.get_primary_auth_time()
        current_refresh_time = self.class_under_test.get_refresh_auth_time()
        current_access_token = self.class_under_test.get_access_token()
        self.class_under_test.refresh_access_token()
        new_auth_time = self.class_under_test.get_primary_auth_time()
        new_access_token = self.class_under_test.get_access_token()
        new_refresh_time = self.class_under_test.get_refresh_auth_time()
        self.assertLess(current_auth_time, new_auth_time)
        self.assertEqual(current_refresh_time, new_refresh_time)
        self.assertNotEqual(current_access_token, new_access_token)

    def test_get_accounts(self):
        self.class_under_test.get_accounts()
        self.assertIsNotNone(self.class_under_test.account_data)
        self.assertGreater(len(self.class_under_test.account_data), 0)

    def test_parse_accounts(self):
        accounts_list = self.class_under_test.parse_accounts()
        self.assertEqual(accounts_list.shape, (3, 8))
        self.assertIn('currentBalances_cashBalance', accounts_list.columns)
        self.assertIn('currentBalances_equity', accounts_list.columns)

    def test_get_positions(self):
        self.class_under_test.get_positions()
        self.assertIsNotNone(self.class_under_test.positions_data)
        self.assertGreater(len(self.class_under_test.positions_data), 0)

    def test_parse_portfolios_list(self):
        portfolio_list = self.class_under_test.parse_portfolios_list()
        self.assertGreater(len(portfolio_list), 0)
        self.assertEqual(portfolio_list.columns.shape, (15,))
        self.assertIn('assetType', portfolio_list.columns)
        self.assertIn('cusip', portfolio_list.columns)
        self.assertIn('marketValue', portfolio_list.columns)
        self.assertIn('longQuantity', portfolio_list.columns)
        self.assertIn('type', portfolio_list.columns)

    def test_refresh_data(self):
        self.class_under_test.account_data = None
        self.class_under_test.positions_data = None
        self.class_under_test.refresh_data()
        self.assertIsNotNone(self.class_under_test.account_data)
        self.assertIsNotNone(self.class_under_test.positions_data)

    def test_create_market_order(self):
        order = amc.create_market_order(
            'TEST_ACCOUNT',
            'AAPL',
            asset_type='EQUITY',
            quantity=1,
            instruction='SELL',
            session='NORMAL',
            duration='DAY')

        self.assertEqual('TEST_ACCOUNT', order['account'])
        self.assertEqual('AAPL', order['symbol'])
        self.assertEqual('EQUITY', order['asset_type'])
        self.assertEqual(1, order['quantity'])
        self.assertEqual('NORMAL', order['session'])
        self.assertEqual('DAY', order['duration'])
        self.assertEqual('MARKET', order['order_type'])
        self.assertNotIn('price', order)

    def test_create_limit_order(self):
        order = amc.create_limit_order(
            'TEST_ACCOUNT',
            'AAPL',
            asset_type='EQUITY',
            quantity=1,
            instruction='SELL',
            session='NORMAL',
            duration='DAY',
            price=1.0)

        self.assertEqual('TEST_ACCOUNT', order['account'])
        self.assertEqual('AAPL', order['symbol'])
        self.assertEqual('EQUITY', order['asset_type'])
        self.assertEqual(1, order['quantity'])
        self.assertEqual('NORMAL', order['session'])
        self.assertEqual('DAY', order['duration'])
        self.assertEqual('LIMIT', order['order_type'])
        self.assertEqual(1.0, order['price'])

    def test_place_saved_order(self):
        self.class_under_test.get_accounts()
        quotes = self.class_under_test.get_quotes(['AAPL'])
        account = self.class_under_test.unmask_account(TEST_MASKED_ACCOUNT)
        order = amc.create_limit_order(
            account,
            'AAPL',
            asset_type='EQUITY',
            quantity=1,
            instruction='SELL',
            session='NORMAL',
            duration='DAY',
            price=quotes.loc['AAPL'].askPrice)

        existing_saved_orders = self.class_under_test.get_saved_orders(account)
        response = self.class_under_test.place_order(order, saved=True)
        self.assertIsNotNone(response)
        self.assertEqual(200, response)
        new_saved_orders = self.class_under_test.get_saved_orders(account)
        self.assertTrue(len(existing_saved_orders) < len(new_saved_orders))

    def test_get_saved_orders(self):
        self.class_under_test.get_accounts()
        quotes = self.class_under_test.get_quotes(['AAPL'])
        account = self.class_under_test.unmask_account(TEST_MASKED_ACCOUNT)
        order = amc.create_limit_order(
            account,
            'AAPL',
            asset_type='EQUITY',
            quantity=1,
            instruction='SELL',
            session='NORMAL',
            duration='DAY',
            price=quotes.loc['AAPL'].askPrice)
        self.class_under_test.place_order(order, saved=True)
        saved_orders = self.class_under_test.get_saved_orders(account)
        self.assertIsNotNone(saved_orders)

    def test_remove_saved_order(self):
        self.class_under_test.get_accounts()
        quotes = self.class_under_test.get_quotes(['AAPL'])
        account = self.class_under_test.unmask_account(TEST_MASKED_ACCOUNT)
        order = amc.create_limit_order(
            account,
            'AAPL',
            asset_type='EQUITY',
            quantity=1,
            instruction='SELL',
            session='NORMAL',
            duration='DAY',
            price=quotes.loc['AAPL'].askPrice)
        self.class_under_test.place_order(order, saved=True)
        saved_orders = self.class_under_test.get_saved_orders(account)
        saved_order_ids = saved_orders.index.tolist()
        for order_id in saved_order_ids:
            self.class_under_test.remove_saved_order(account, order_id)
        saved_orders = self.class_under_test.get_saved_orders(account)
        self.assertIsNone(saved_orders)


