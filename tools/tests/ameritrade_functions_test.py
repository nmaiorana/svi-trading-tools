import unittest
import tools.ameritrade_functions as amc
import os
import json
import configparser

TEST_USER = 'TEST_USER'
TEST_PW = 'TEST_PW'
TEST_CLIENT_ID = 'TEST_CLIENT_ID'
TEST_CONFIG_PATH = 'test_config/td_config.ini'

# # Unit Tests

# ## Test Configuration
# 
# Configure the Ameritrade client to use the username, password and client id stored in environment variables.


class TestConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = configparser.ConfigParser()
        config.read(TEST_CONFIG_PATH)
        test_config = config[amc.CONFIG_SECTION]
        cls.test_config = test_config
        os.environ[cls.test_config[amc.ENV_USER_VARIABLE]] = TEST_USER
        os.environ[cls.test_config[amc.ENV_PW_VARIABLE]] = TEST_PW
        os.environ[cls.test_config[amc.ENV_CLIENT_ID_VARIABLE]] = TEST_CLIENT_ID

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
        cls.class_under_test.authenticate()

    def test_get_authorization(self):
        self.assertIsNotNone(self.class_under_test.get_authorization())
        
    def test_authentication(self):
        self.assertGreater(len(self.class_under_test.authorization), 0)
        self.assertIsNotNone(self.class_under_test.get_primary_auth_time())
        self.assertIsNotNone(self.class_under_test.get_refresh_auth_time())
        
    def test_get_access_token(self):
        self.assertIsNotNone(self.class_under_test.get_access_token())
        
        # Unauthenticated
        with self.assertRaises(RuntimeError) as cm:
            amc.AmeritradeRest().get_access_token()

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
