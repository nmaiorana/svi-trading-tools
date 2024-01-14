import unittest
import logging
from datetime import datetime
from pathlib import Path
import configparser
import pandas as pd
import tools.price_histories_helper as phh

logging.config.fileConfig('./test_config/logging.ini')


TEST_DATA_CSV = 'test_data/price_hist_helper_test_data.parquet'


class PriceHistoriesHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = configparser.ConfigParser()
        config.read('test_config/ai_alpha_config.ini')
        cls.config = config['DEFAULT']
        cls.price_histories_path = Path(TEST_DATA_CSV)

    def test_from_yahoo_finance(self):
        self.price_histories_path.unlink(missing_ok=True)
        start = datetime(year=2019, month=1, day=1)
        end = datetime(year=2020, month=1, day=1)
        test_data_df = phh.from_yahoo_finance(symbols=['AAPL', 'GOOG'],
                                              storage_path=self.price_histories_path,
                                              start=start, end=end, reload=True)
        self.assertIsNotNone(test_data_df)
        self.assertSetEqual({'AAPL', 'GOOG'}, set(test_data_df.columns.get_level_values('Symbols')))
        self.assertIsInstance(test_data_df.index, pd.DatetimeIndex)
        self.assertTrue(self.price_histories_path.exists())

    def test_from_yahoo_finance_from_file(self):
        self.price_histories_path.unlink(missing_ok=True)
        start = datetime(year=2019, month=1, day=1)
        end = datetime(year=2020, month=1, day=1)
        test_data_df = phh.from_yahoo_finance(symbols=['AAPL', 'GOOG'],
                                              storage_path=self.price_histories_path,
                                              start=start, end=end, reload=True)
        self.assertIsNotNone(test_data_df)
        self.assertSetEqual({'AAPL', 'GOOG'}, set(test_data_df.columns.get_level_values('Symbols')))
        self.assertIsInstance(test_data_df.index, pd.DatetimeIndex)
        self.assertTrue(self.price_histories_path.exists())

        # Now reload from file
        test_data_df = phh.from_yahoo_finance(symbols=['AAPL', 'GOOG'],
                                              storage_path=self.price_histories_path,
                                              start=start, end=end, reload=False)
        self.assertIsNotNone(test_data_df)
        self.assertSetEqual({'AAPL', 'GOOG'}, set(test_data_df.columns.get_level_values('Symbols')))
        self.assertIsInstance(test_data_df.index, pd.DatetimeIndex)
        self.assertTrue(self.price_histories_path.exists())

    def test_from_yahoo_finance_period(self):
        self.price_histories_path.unlink(missing_ok=True)
        period = '1y'
        test_data_df = phh.from_yahoo_finance(symbols=['AAPL', 'GOOG'],
                                              storage_path=self.price_histories_path,
                                              period=period, reload=False)
        self.assertIsNotNone(test_data_df)
        self.assertSetEqual({'AAPL', 'GOOG'}, set(test_data_df.columns.get_level_values('Symbols')))
        self.assertIsInstance(test_data_df.index, pd.DatetimeIndex)
        self.assertTrue(self.price_histories_path.exists())

    def test_download_histories_and_adjust(self):
        start = datetime(year=2019, month=1, day=1)
        end = datetime(year=2020, month=1, day=1)
        test_data_df = phh.download_histories_and_adjust(symbols=['AAPL', 'GOOG'], start=start, end=end)
        self.assertIsNotNone(test_data_df)
        self.assertSetEqual({'AAPL', 'GOOG'}, set(test_data_df.columns.get_level_values('Symbols')))
        self.assertIsInstance(test_data_df.index, pd.DatetimeIndex)


if __name__ == '__main__':
    unittest.main()
