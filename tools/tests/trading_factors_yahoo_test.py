import unittest
import pandas as pd
import os.path
from pathlib import Path
import pandas_datareader as pdr
from datetime import datetime
import tools.trading_factors_yahoo as alpha_factors
import ssl

# This is used to get s&p 500 data. Without it, we get cert errors
ssl._create_default_https_context = ssl._create_unverified_context

# Make sure we have a data directory
Path('./test_data').mkdir(parents=True, exist_ok=True)

# # Test Harness

test_data_file = './test_data/test_data.csv'
if os.path.exists(test_data_file):
    test_data_df = pd.read_csv(test_data_file, header=[0, 1], index_col=[0], parse_dates=True, low_memory=False)
else:
    start = datetime(year=2019, month=1, day=1)
    end = datetime(year=2020, month=1, day=1)

    yahoo_reader = pdr.yahoo.daily.YahooDailyReader(symbols=['AAPL', 'GOOG'], start=start, end=end, adjust_price=True,
                                                    interval='d', get_actions=False, adjust_dividends=True)
    test_data_df = yahoo_reader.read()
    yahoo_reader.close()
    test_data_df.to_csv(test_data_file, index=True)

test_snp_500_stocks_file = './test_data/snp500.csv'
if os.path.exists(test_snp_500_stocks_file):
    snp_500_stocks = pd.read_csv(test_snp_500_stocks_file, index_col=[0], low_memory=False)
else:
    snp_500_stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                                  header=0,
                                  attrs={'id': 'constituents'},
                                  index_col='Symbol')[0]
    snp_500_stocks.to_csv(test_snp_500_stocks_file)



class TestFactorData(unittest.TestCase):
    def test_init(self):
        class_under_test = alpha_factors.FactorData(test_data_df)
        self.assertEqual(class_under_test.factor_name, 'Alpha Factor')
        self.assertEqual(len(class_under_test.factor_data.columns), len(test_data_df.columns))

    def test_rank(self):
        class_under_test = alpha_factors.FactorReturns(test_data_df).rank()
        self.assertEqual(class_under_test.factor_data.loc['2019-01-03']['AAPL'], 1.0)
        self.assertEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], 2.0)

    def test_demean(self):
        class_under_test = alpha_factors.FactorReturns(test_data_df).demean()
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-01-03']['AAPL'], -0.035561, places=4)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], 0.0033238, places=4)

    def test_zscore(self):
        class_under_test = alpha_factors.FactorReturns(test_data_df).zscore()
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-01-03']['AAPL'], -1.0, places=4)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], 1.0, places=4)

    def test_smoothed(self):
        class_under_test = alpha_factors.FactorReturns(test_data_df).smoothed(2)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-01-04']['AAPL'], -0.028459, places=4)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], 0.006621, places=4)

    def test_for_al(self):
        series_data = alpha_factors.FactorReturns(test_data_df).for_al()
        self.assertEqual(series_data.name, 'returns_1_day')
        self.assertTupleEqual(series_data.shape, (502,))
        self.assertAlmostEqual(series_data.loc['2019-12-31', 'AAPL'], 0.007306, places=4)

    def test_open_values(self):
        class_under_test = alpha_factors.OpenPrices(test_data_df)
        self.assertEqual(class_under_test.factor_name, 'open')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-01-02']['AAPL'], 37.488012, places=4)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], 71.226059, places=4)

    def test_close_values(self):
        class_under_test = alpha_factors.ClosePrices(test_data_df)
        self.assertEqual(class_under_test.factor_name, 'close')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-01-02']['AAPL'], 38.221363, places=4)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], 72.139938, places=4)

    def test_volume_values(self):
        class_under_test = alpha_factors.Volume(test_data_df)
        self.assertEqual(class_under_test.factor_name, 'volume')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-01-02']['AAPL'], 148158800.0, places=4)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], 100805600.0, places=4)

    def test_top(self):
        class_under_test = alpha_factors.Volume(test_data_df).top(1)
        self.assertEqual(len(class_under_test), 1)
        self.assertEqual(class_under_test[0], 'AAPL')

    def test_daily_dollar_volume(self):
        class_under_test = alpha_factors.DailyDollarVolume(test_data_df)
        self.assertEqual(class_under_test.factor_name, 'daily_dollar_volume')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-01-02']['AAPL'], 5662831286.463928, places=2)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], 7272109769.787598, places=2)

    def test_average_dollar_volume(self):
        class_under_test = alpha_factors.AverageDollarVolume(test_data_df, 5)
        self.assertEqual(class_under_test.factor_name, 'average_dollar_volume_5_day')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-01-08']['AAPL'], 8095402562.039185, places=2)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], 7604826388.565674, places=2)

    def test_returns(self):
        class_under_test = alpha_factors.FactorReturns(test_data_df)
        self.assertEqual(class_under_test.factor_name, 'returns_1_day')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertTupleEqual(class_under_test.factor_data.shape, (252, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], 0.007306, places=4)

    def test_momentum(self):
        class_under_test = alpha_factors.FactorMomentum(test_data_df, 5)
        self.assertEqual(class_under_test.factor_name, 'momentum_5_day')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertTupleEqual(class_under_test.factor_data.shape, (252, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], 0.033978, places=4)

    def test_meanreversion(self):
        class_under_test = alpha_factors.FactorMeanReversion(test_data_df, 5)
        self.assertEqual(class_under_test.factor_name, 'mean_reversion_5_day_logret')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertTupleEqual(class_under_test.factor_data.shape, (252, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], -0.033978, places=4)

    def test_close_to_open(self):
        class_under_test = alpha_factors.CloseToOpen(test_data_df)
        self.assertEqual(class_under_test.factor_name, 'close_to_open')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertTupleEqual(class_under_test.factor_data.shape, (252, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['AAPL'], -0.005454, places=4)

    def test_trailing_overnight_returns(self):
        class_under_test = alpha_factors.TrailingOvernightReturns(test_data_df, 10)
        self.assertEqual(class_under_test.factor_name, 'trailing_overnight_returns_10_day')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertTupleEqual(class_under_test.factor_data.shape, (252, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['AAPL'], 0.00963419, places=4)

    def test_annualized_volatility(self):
        class_under_test = alpha_factors.AnnualizedVolatility(test_data_df, 20)
        self.assertEqual(class_under_test.factor_name, 'annualzed_volatility_20_day')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertTupleEqual(class_under_test.factor_data.shape, (213, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['AAPL'], 0.3957571629968691, places=4)

    def test_market_dispersion(self):
        class_under_test = alpha_factors.MarketDispersion(test_data_df, 20)
        self.assertEqual(class_under_test.factor_name, 'market_dispersion_20_day')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertTupleEqual(class_under_test.factor_data.shape, (251, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['AAPL'], 0.0042166016735185, places=4)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['GOOG'], 0.0042166016735185, places=4)

    def test_market_volatility(self):
        class_under_test = alpha_factors.MarketVolatility(test_data_df, 20)
        self.assertEqual(class_under_test.factor_name, 'market_volatility_20_day')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertTupleEqual(class_under_test.factor_data.shape, (252, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['AAPL'], 0.10326828506031509, places=4)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['GOOG'], 0.10326828506031509, places=4)

    def test_factor_return_quantiles(self):
        class_under_test = alpha_factors.FactorReturnQuantiles(test_data_df, 5)
        self.assertEqual(class_under_test.factor_name, 'logret_1_day_5_quantiles')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertTupleEqual(class_under_test.factor_data.shape, (252, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['AAPL'], 4.0, places=4)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['GOOG'], 0.0, places=4)

    def test_get_sector_helper(self):
        class_under_test = alpha_factors.get_sector_helper(snp_500_stocks, 'GICS Sector', ['AAPL', 'GOOG'])
        self.assertEqual(class_under_test['Communication Services'], ['GOOG'])
        self.assertEqual(class_under_test['Information Technology'], ['AAPL'])

    def test_filter_price_histories(self):
        class_under_test = set(
            alpha_factors.filter_price_histories(test_data_df, ['AAPL']).columns.get_level_values('Symbols').tolist())
        self.assertEqual(class_under_test.pop(), 'AAPL')

    def test_prepare_alpha_lens_factor_data(self):
        alpha_factors_list = [
            alpha_factors.AverageDollarVolume(test_data_df, 5).for_al(),
            alpha_factors.FactorReturns(test_data_df).for_al()
        ]
        all_factors = pd.concat(alpha_factors_list, axis=1)
        pricing = test_data_df.Close
        clean_factor_data, unixt_factor_data = alpha_factors.prepare_alpha_lens_factor_data(all_factors, pricing)
        self.assertEqual(494, len(list(clean_factor_data.values())[0]))
        self.assertEqual(494, len(list(unixt_factor_data.values())[0]))

    def test_eval_factor_and_add(self) -> None:
        factors_list = []
        pricing = test_data_df.Close
        factor_to_eval = alpha_factors.AnnualizedVolatility(test_data_df, 10).rank().zscore()
        alpha_factors.eval_factor_and_add(factors_list, factor_to_eval, pricing, 0.0)
        self.assertEqual(len(factors_list), 1)

    def test_eval_factor_and_add_no_add(self) -> None:
        factors_list = []
        pricing = test_data_df.Close
        factor_to_eval = alpha_factors.AnnualizedVolatility(test_data_df, 10).rank().zscore()
        alpha_factors.eval_factor_and_add(factors_list, factor_to_eval, pricing, 100.0)
        self.assertEqual(len(factors_list), 0)

    def test_get_factor_returns(self) -> None:
        alpha_factors_list = [
            alpha_factors.AverageDollarVolume(test_data_df, 5).for_al(),
            alpha_factors.FactorReturns(test_data_df).for_al()
        ]
        all_factors = pd.concat(alpha_factors_list, axis=1)
        pricing = test_data_df.Close
        clean_factor_data, _ = alpha_factors.prepare_alpha_lens_factor_data(all_factors, pricing)
        factor_returns_data = alpha_factors.get_factor_returns(clean_factor_data)
        self.assertEqual(247, len(factor_returns_data))
        self.assertAlmostEqual(factor_returns_data.iloc[0][0], 0.0092433, places=4)
        self.assertAlmostEqual(factor_returns_data.iloc[-1][0], 0.0033237, places=4)

    def test_compute_sharpe_ratio(self) -> None:
        pricing = test_data_df.Close
        factor_data = alpha_factors.AverageDollarVolume(test_data_df, 5).for_al()
        clean_factor_data, _ = alpha_factors.prepare_alpha_lens_factor_data(factor_data.to_frame().copy(), pricing)
        factor_returns = alpha_factors.get_factor_returns(clean_factor_data)
        sharpe_ratio = alpha_factors.compute_sharpe_ratio(factor_returns, frequency='daily')['Sharpe Ratio'].values[0]
        self.assertEqual(2.1, sharpe_ratio)
        sharpe_ratio = alpha_factors.compute_sharpe_ratio(factor_returns, frequency='monthly')['Sharpe Ratio'].values[0]
        self.assertEqual(0.46, sharpe_ratio)
        sharpe_ratio = alpha_factors.compute_sharpe_ratio(factor_returns, frequency='yearly')['Sharpe Ratio'].values[0]
        self.assertEqual(0.13, sharpe_ratio)