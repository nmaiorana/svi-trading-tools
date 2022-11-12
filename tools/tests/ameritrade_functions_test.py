import unittest
import tools.ameritrade_functions as amc
import os
import json

custom_username_env = 'maiotradeuser'
custom_pw_env = 'maiotradepw'
custom_clientid_env = 'maiotradeclientid'

# # Unit Tests

# ## Test Configuration
# 
# Configure the Ameritrade client to use the username, password and client id stored in environment variables.

os.environ['ameritradeuser'] = 'test_user'
os.environ['ameritradepw'] = 'test_pw'
os.environ['ameritradeclientid'] = 'client_id'


class TestConfiguration(unittest.TestCase):
    def test_config_default(self):
        class_under_test = amc.AmeritradeRest()
        self.assertEqual(class_under_test.username, 'test_user')
        self.assertEqual(class_under_test.password, 'test_pw')
        self.assertEqual(class_under_test.client_id, 'client_id')  
        self.assertEqual(class_under_test.consumer_key, 'client_id'+'@AMER.OAUTHAP')
    
    def test_config_params(self):
        class_under_test = amc.AmeritradeRest('ameritradeuser', 'ameritradepw', 'ameritradeclientid')
        self.assertEqual(class_under_test.username, 'test_user')
        self.assertEqual(class_under_test.password, 'test_pw')
        self.assertEqual(class_under_test.client_id, 'client_id')
        self.assertEqual(class_under_test.consumer_key, 'client_id'+'@AMER.OAUTHAP')

# ## Account Level Functions


class TestAccountLevelFunctions(unittest.TestCase):
    def test_mask_account(self):
        class_under_test = amc.AmeritradeRest(custom_username_env, custom_pw_env, custom_clientid_env)
        self.assertEqual(class_under_test.mask_account('123456789'), '#---6789')
        class_under_test.account_mask = "*****"
        self.assertEqual(class_under_test.mask_account('12345678'), '*****5678')
        
    def test_unmasked_accounts(self):
        class_under_test = amc.AmeritradeRest(custom_username_env, custom_pw_env, custom_clientid_env)
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
        cls.class_under_test = amc.AmeritradeRest(custom_username_env, custom_pw_env, custom_clientid_env)
        
    def test_get_daily_price_history(self):
        price_histories = self.class_under_test.get_daily_price_history('AAPL', '2022-01-03')
        self.assertEqual(len(price_histories), 253)
        self.assertEqual(price_histories.date.min().strftime('%Y-%m-%d'), '2021-01-04')
        self.assertEqual(price_histories.date.max().strftime('%Y-%m-%d'), '2022-01-03')
        self.assertEqual(price_histories.ticker.unique()[0], 'AAPL')
        
    def test_get_price_histories(self):
        price_histories = self.class_under_test.get_price_histories(['AAPL', 'GOOG'], '2022-01-03', silent=True)
        print(price_histories)
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
        cls.class_under_test = amc.AmeritradeRest(custom_username_env, custom_pw_env, custom_clientid_env)
        cls.class_under_test.authenticate()
        
    def test_authentication(self):
        self.assertGreater(len(self.class_under_test.authorization), 0)
        
    def test_get_access_token(self):
        self.assertIsNotNone(self.class_under_test.get_access_token())
        
        # Unauthenticated
        with self.assertRaises(RuntimeError) as cm:
            amc.AmeritradeRest(custom_username_env, custom_pw_env, custom_clientid_env).get_access_token()

    def test_get_accounts(self):
        self.class_under_test.get_accounts()
        self.assertIsNotNone(self.class_under_test.account_data)
        self.assertGreater(len(self.class_under_test.account_data), 0)

    def test_parse_accounts(self):
        accounts_list = self.class_under_test.parse_accounts()
        self.assertEquals(accounts_list.shape, (3, 8))
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


class TestAccountFunctions(unittest.TestCase):
    masked_account_1 = "#---1111"
    masked_account_2 = "#---2222"
    masked_account_3 = "#---3333"

    @classmethod
    def setUpClass(cls) -> None:
        cls.class_under_test = amc.AmeritradeRest(custom_username_env, custom_pw_env, custom_clientid_env)
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
        self.assertEquals(accounts_list.shape, (3, 8))
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

