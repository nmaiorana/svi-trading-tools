import unittest
import tools.ameritrade_functions as amc
import os

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
    
    def test_config_parms(self):
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


# In[4]:


TestAccountLevelFunctions().test_unmasked_accounts()


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
    def test_parse_accounts(self):
        self.assertGreater(len(self.class_under_test.parse_accounts()), 0)
        
    def test_get_positions(self):
        self.assertGreater(len(self.class_under_test.get_positions()), 0)
        
    def test_parse_portfolios_list(self):
        self.assertGreater(len(self.class_under_test.parse_portfolios_list()), 0)