import unittest
from datetime import datetime
from pathlib import Path
import configparser

import pandas as pd

import tools.price_histories_helper as phh

TEST_DATA_DIR = 'test_data'

TEST_DATA_CSV = 'price_hist_helper_test_data.csv'


class PriceHistoriesHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_config = configparser.ConfigParser()
        cls.test_config['DEFAULT'] = {
            phh.DATA_DIRECTORY_KEY: './' + TEST_DATA_DIR,
            phh.HISTORIES_FILE_NAME_KEY: TEST_DATA_CSV,
            'NumberOfYearsPriceHistories': 1
        }

    def test_default_histories_path(self):
        hist_path = phh.default_histories_path(self.test_config['DEFAULT'])
        self.assertEqual(('test_data', TEST_DATA_CSV), hist_path.parts)

    def test_default_snp500_path(self):
        hist_path = phh.default_histories_path(self.test_config['DEFAULT'])
        snp_path = phh.default_snp500_path(hist_path)
        self.assertEqual(('test_data', phh.DEFAULT_SNP500_FILE), snp_path.parts)

    def test_default_snp500_path_config(self):
        snp_path = phh.default_snp500_path_config(self.test_config['DEFAULT'])
        self.assertEqual(('test_data', phh.DEFAULT_SNP500_FILE), snp_path.parts)

    def test_from_yahoo_finance(self):
        start = datetime(year=2019, month=1, day=1)
        end = datetime(year=2020, month=1, day=1)
        test_data_df = phh.from_yahoo_finance(symbols=['AAPL', 'GOOG'],
                                              storage_path=Path('test_data/test_data.csv'),
                                              start=start, end=end, reload=True)
        self.assertIsNotNone(test_data_df)
        self.assertSetEqual(set(['AAPL', 'GOOG']), set(test_data_df.Close.columns.values))
        self.assertIsInstance(test_data_df.index, pd.DatetimeIndex)
        test_data_df = phh.from_yahoo_finance(symbols=['AAPL', 'GOOG'],
                                              storage_path=Path('test_data/test_data.csv'),
                                              start=start, end=end, reload=False)
        self.assertIsNotNone(test_data_df)
        self.assertSetEqual(set(['AAPL', 'GOOG']), set(test_data_df.Close.columns.values))
        self.assertIsInstance(test_data_df.index, pd.DatetimeIndex)

    def test_from_yahoo_finance_config(self):
        test_data_df = phh.from_yahoo_finance_config(self.test_config['DEFAULT'], ['AAPL', 'GOOG'], reload=True)
        self.assertIsNotNone(test_data_df)
        self.assertIsInstance(test_data_df.index, pd.DatetimeIndex)
        test_data_df = phh.from_yahoo_finance_config(self.test_config['DEFAULT'], ['AAPL', 'GOOG'], reload=False)
        self.assertIsNotNone(test_data_df)
        self.assertIsInstance(test_data_df.index, pd.DatetimeIndex)

    def test_download_histories_and_adjust(self):
        start = datetime(year=2019, month=1, day=1)
        end = datetime(year=2020, month=1, day=1)
        test_data_df = phh.download_histories_and_adjust(symbols=['AAPL', 'GOOG'],
                                                         start=start, end=end)
        self.assertIsNotNone(test_data_df)
        self.assertSetEqual(set(['AAPL', 'GOOG']), set(test_data_df.Close.columns.values))
        self.assertIsInstance(test_data_df.index, pd.DatetimeIndex)

    def test_load_snp500_symbols(self):
        snp_500_stocks = phh.load_snp500_symbols(Path('./test_data/snp500.csv'))
        self.assertIsNotNone(snp_500_stocks)
        self.assertIsInstance(snp_500_stocks, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
