import unittest
from datetime import datetime
from pathlib import Path
import configparser

import tools.price_histories_helper as phh


class PriceHistoriesHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_config = configparser.ConfigParser()
        cls.test_config['DEFAULT'] = {
            phh.DATA_DIRECTORY: './test_data',
            phh.HISTORIES_FILE_NAME: 'snp500.csv'
        }

    def test_default_histories_path(self):
        path_name = phh.default_histories_path(self.test_config['DEFAULT'])
        self.assertEqual(('test_data', 'snp500.csv'), path_name.parts)  # add assertion here

    def test_from_yahoo_finance(self):
        start = datetime(year=2019, month=1, day=1)
        end = datetime(year=2020, month=1, day=1)
        test_data_df = phh.from_yahoo_finance(symbols=['AAPL', 'GOOG'],
                                              storage_path=Path('test_data/test_data.csv'),
                                              start=start, end=end, reload=True)
        self.assertIsNotNone(test_data_df)

    def test_load_snp500_symbols(self):
        snp_500_stocks = phh.load_snp500_symbols(Path('./test_data/snp500.csv'))
        self.assertIsNotNone(snp_500_stocks)


if __name__ == '__main__':
    unittest.main()
