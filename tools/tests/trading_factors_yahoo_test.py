import unittest
import pandas as pd
import os.path
from pathlib import Path
import pandas_datareader as pdr
import yfinance as yf
from datetime import datetime
import tools.trading_factors_yahoo as alpha_factors
import ssl

# This is used to get s&p 500 data. Without it, we get cert errors
ssl._create_default_https_context = ssl._create_unverified_context

# Make sure we have a data directory
Path('test_data').mkdir(parents=True, exist_ok=True)

# # Test Harness

test_data_file = 'test_data/test_data.csv'


class TestFactorData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(test_data_file):
            test_data_df = pd.read_csv(test_data_file, header=[0, 1], index_col=[0], parse_dates=True, low_memory=False)
            test_data_df = test_data_df.round(2)
        else:
            start = datetime(year=2019, month=1, day=1)
            end = datetime(year=2020, month=1, day=1)
            '''
            yahoo_reader = pdr.yahoo.daily.YahooDailyReader(symbols=['AAPL', 'GOOG'], start=start, end=end,
                                                            adjust_price=True,
                                                            interval='d', get_actions=False, adjust_dividends=True)
            test_data_df = yahoo_reader.read()
            yahoo_reader.close()
            '''
            test_data_df = yf.download(tickers=['AAPL', 'GOOG'], start=start, end=end,
                                       auto_adjust=True)
            test_data_df.rename_axis(columns=['Attributes', 'Symbols'], inplace=True)
            test_data_df = test_data_df.round(2)
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

        cls.test_data_df = test_data_df
        cls.snp_500_stocks = snp_500_stocks

    def test_init(self):
        class_under_test = alpha_factors.FactorData(self.test_data_df)
        self.assertEqual('Alpha Factor', class_under_test.factor_name)
        self.assertEqual(len(self.test_data_df.columns), len(class_under_test.factor_data.columns))

    def test_rank(self):
        class_under_test = alpha_factors.FactorReturns(self.test_data_df).rank()
        self.assertEqual(1.0, class_under_test.factor_data.loc['2019-01-03']['AAPL'])
        self.assertEqual(2.0, class_under_test.factor_data.loc['2019-12-31']['AAPL'])

    def test_demean(self):
        class_under_test = alpha_factors.FactorReturns(self.test_data_df).demean()
        self.assertAlmostEqual(-0.035561, class_under_test.factor_data.loc['2019-01-03']['AAPL'], places=4)
        self.assertAlmostEqual(0.0033238, class_under_test.factor_data.loc['2019-12-31']['AAPL'], places=4)

    def test_zscore(self):
        class_under_test = alpha_factors.FactorReturns(self.test_data_df).zscore()
        self.assertEqual(-1.0, class_under_test.factor_data.loc['2019-01-03']['AAPL'])
        self.assertEqual(1.0, class_under_test.factor_data.loc['2019-12-31']['AAPL'])

    def test_smoothed(self):
        class_under_test = alpha_factors.FactorReturns(self.test_data_df).smoothed(2)
        self.assertAlmostEqual(-0.028459, class_under_test.factor_data.loc['2019-01-04']['AAPL'], places=2)
        self.assertAlmostEqual(0.006621, class_under_test.factor_data.loc['2019-12-31']['AAPL'], places=2)

    def test_for_al(self):
        series_data = alpha_factors.FactorReturns(self.test_data_df).for_al()
        self.assertEqual('returns_1_day', series_data.name)
        self.assertTupleEqual((502,), series_data.shape)
        self.assertAlmostEqual(0.007306, series_data.loc['2019-12-31', 'AAPL'], places=4)

    def test_open_values(self):
        class_under_test = alpha_factors.OpenPrices(self.test_data_df)
        self.assertEqual('open', class_under_test.factor_name)
        self.assertEqual('AAPL', class_under_test.factor_data.columns[0])
        self.assertAlmostEqual(37.37401281863855, class_under_test.factor_data.loc['2019-01-02']['AAPL'], places=2)
        self.assertAlmostEqual(71.00947003250288, class_under_test.factor_data.loc['2019-12-31']['AAPL'], places=2)

    def test_close_values(self):
        class_under_test = alpha_factors.ClosePrices(self.test_data_df)
        self.assertEqual('close', class_under_test.factor_name)
        self.assertEqual('AAPL', class_under_test.factor_data.columns[0])
        self.assertAlmostEqual(38.10513305664063, class_under_test.factor_data.loc['2019-01-02']['AAPL'], places=2)
        self.assertAlmostEqual(71.92057037353516, class_under_test.factor_data.loc['2019-12-31']['AAPL'], places=2)

    def test_volume_values(self):
        class_under_test = alpha_factors.Volume(self.test_data_df)
        self.assertEqual('volume', class_under_test.factor_name)
        self.assertEqual('AAPL', class_under_test.factor_data.columns[0])
        self.assertEqual(148158800.0, class_under_test.factor_data.loc['2019-01-02']['AAPL'])
        self.assertEqual(100805600.0, class_under_test.factor_data.loc['2019-12-31']['AAPL'])

    def test_top(self):
        class_under_test = alpha_factors.Volume(self.test_data_df).top(1)
        self.assertEqual(1, len(class_under_test))
        self.assertEqual('AAPL', class_under_test[0])

    def test_daily_dollar_volume(self):
        class_under_test = alpha_factors.DailyDollarVolume(self.test_data_df)
        self.assertEqual(class_under_test.factor_name, 'daily_dollar_volume')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')

        close_values = alpha_factors.ClosePrices(self.test_data_df).factor_data
        volume_values = alpha_factors.Volume(self.test_data_df).factor_data

        for test_date in self.test_data_df.index:
            expected_value = close_values.loc[test_date]['AAPL'] * volume_values.loc[test_date]['AAPL']
            self.assertAlmostEqual(expected_value, class_under_test.factor_data.loc[test_date]['AAPL'], places=2)

    def test_average_dollar_volume(self):
        class_under_test = alpha_factors.AverageDollarVolume(self.test_data_df, 5)
        self.assertEqual(class_under_test.factor_name, 'average_dollar_volume_5_day')
        self.assertEqual(class_under_test.factor_data.columns[0], 'AAPL')
        self.assertTrue(pd.isna(class_under_test.factor_data.loc['2019-01-02']['AAPL']))
        self.assertTrue(pd.isna(class_under_test.factor_data.loc['2019-01-03']['AAPL']))
        self.assertTrue(pd.isna(class_under_test.factor_data.loc['2019-01-04']['AAPL']))
        self.assertTrue(pd.isna(class_under_test.factor_data.loc['2019-01-07']['AAPL']))
        self.assertAlmostEqual(8070720489.6, class_under_test.factor_data.loc['2019-01-08']['AAPL'], places=2)
        self.assertAlmostEqual(7581673369.6, class_under_test.factor_data.loc['2019-12-31']['AAPL'], places=2)

    def test_returns(self):
        class_under_test = alpha_factors.FactorReturns(self.test_data_df)
        self.assertEqual('returns_1_day', class_under_test.factor_name)
        self.assertEqual('AAPL', class_under_test.factor_data.columns[0])
        self.assertTupleEqual((252, 2), class_under_test.factor_data.shape)
        self.assertAlmostEqual(0.007306, class_under_test.factor_data.loc['2019-12-31']['AAPL'], places=4)

    def test_momentum(self):
        class_under_test = alpha_factors.FactorMomentum(self.test_data_df, 5)
        self.assertEqual('momentum_5_day', class_under_test.factor_name)
        self.assertEqual('AAPL', class_under_test.factor_data.columns[0])
        self.assertTupleEqual(class_under_test.factor_data.shape, (252, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], 0.033978, places=2)

    def test_mean_reversion(self):
        class_under_test = alpha_factors.FactorMeanReversion(self.test_data_df, 5)
        self.assertEqual(class_under_test.factor_name, 'mean_reversion_5_day_logret')
        self.assertEqual('AAPL', class_under_test.factor_data.columns[0])
        self.assertTupleEqual(class_under_test.factor_data.shape, (252, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-31']['AAPL'], -0.033978, places=2)

    def test_close_to_open(self):
        class_under_test = alpha_factors.CloseToOpen(self.test_data_df)
        self.assertEqual(class_under_test.factor_name, 'close_to_open')
        self.assertEqual('AAPL', class_under_test.factor_data.columns[0])
        self.assertTupleEqual(class_under_test.factor_data.shape, (252, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['AAPL'], -0.005454, places=2)

    def test_trailing_overnight_returns(self):
        class_under_test = alpha_factors.TrailingOvernightReturns(self.test_data_df, 10)
        self.assertEqual(class_under_test.factor_name, 'trailing_overnight_returns_10_day')
        self.assertEqual('AAPL', class_under_test.factor_data.columns[0])
        self.assertTupleEqual(class_under_test.factor_data.shape, (252, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['AAPL'], 0.009688, places=2)

    def test_annualized_volatility(self):
        class_under_test = alpha_factors.AnnualizedVolatility(self.test_data_df, 20)
        self.assertEqual(class_under_test.factor_name, 'annualized_volatility_20_day')
        self.assertEqual('AAPL', class_under_test.factor_data.columns[0])
        self.assertTupleEqual(class_under_test.factor_data.shape, (213, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['AAPL'], 0.3957571629968691, places=4)

    def test_market_dispersion(self):
        class_under_test = alpha_factors.MarketDispersion(self.test_data_df, 20)
        self.assertEqual(class_under_test.factor_name, 'market_dispersion_20_day')
        self.assertEqual('AAPL', class_under_test.factor_data.columns[0])
        self.assertTupleEqual(class_under_test.factor_data.shape, (251, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['AAPL'], 0.0042166016735185, places=4)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['GOOG'], 0.0042166016735185, places=4)

    def test_market_volatility(self):
        class_under_test = alpha_factors.MarketVolatility(self.test_data_df, 20)
        self.assertEqual(class_under_test.factor_name, 'market_volatility_20_day')
        self.assertEqual('AAPL', class_under_test.factor_data.columns[0])
        self.assertTupleEqual(class_under_test.factor_data.shape, (252, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['AAPL'], 0.10326828506031509, places=4)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['GOOG'], 0.10326828506031509, places=4)

    def test_factor_return_quantiles(self):
        class_under_test = alpha_factors.FactorReturnQuantiles(self.test_data_df, 5)
        self.assertEqual(class_under_test.factor_name, 'logret_1_day_5_quantiles')
        self.assertEqual('AAPL', class_under_test.factor_data.columns[0])
        self.assertTupleEqual(class_under_test.factor_data.shape, (252, 2))
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['AAPL'], 4.0, places=4)
        self.assertAlmostEqual(class_under_test.factor_data.loc['2019-12-30']['GOOG'], 0.0, places=4)

    def test_get_sector_helper(self):
        class_under_test = alpha_factors.get_sector_helper(self.snp_500_stocks, 'GICS Sector', ['AAPL', 'GOOG'])
        self.assertEqual(class_under_test['Communication Services'], ['GOOG'])
        self.assertEqual(class_under_test['Information Technology'], ['AAPL'])

    def test_filter_price_histories(self):
        class_under_test = set(
            alpha_factors.filter_price_histories(self.test_data_df, ['AAPL']).columns.get_level_values(
                'Symbols').tolist())
        self.assertEqual(class_under_test.pop(), 'AAPL')

    def test_prepare_alpha_lens_factor_data(self):
        alpha_factors_list = [
            alpha_factors.AverageDollarVolume(self.test_data_df, 5).for_al(),
            alpha_factors.FactorReturns(self.test_data_df).for_al()
        ]
        all_factors = pd.concat(alpha_factors_list, axis=1)
        pricing = self.test_data_df.Close
        clean_factor_data, unixt_factor_data = alpha_factors.prepare_alpha_lens_factor_data(all_factors, pricing)
        self.assertEqual(494, len(list(clean_factor_data.values())[0]))
        self.assertEqual(494, len(list(unixt_factor_data.values())[0]))

    def test_eval_factor_and_add(self) -> None:
        factors_list = []
        pricing = self.test_data_df.Close
        factor_to_eval = alpha_factors.AnnualizedVolatility(self.test_data_df, 10).rank().zscore()
        alpha_factors.eval_factor_and_add(factors_list, factor_to_eval, pricing, 0.0)
        self.assertEqual(1, len(factors_list))

    def test_eval_factor_and_add_no_add(self) -> None:
        factors_list = []
        pricing = self.test_data_df.Close
        factor_to_eval = alpha_factors.AnnualizedVolatility(self.test_data_df, 10).rank().zscore()
        alpha_factors.eval_factor_and_add(factors_list, factor_to_eval, pricing, 100.0)
        self.assertEqual(0, len(factors_list))

    def test_get_factor_returns(self) -> None:
        pricing = self.test_data_df.Close
        factor_data = alpha_factors.AverageDollarVolume(self.test_data_df, 5).for_al()
        clean_factor_data, _ = alpha_factors.prepare_alpha_lens_factor_data(factor_data.to_frame().copy(), pricing)
        factor_returns_data = alpha_factors.get_factor_returns(clean_factor_data)
        self.assertEqual(247, len(factor_returns_data))
        self.assertAlmostEqual(0.0092433, factor_returns_data.iloc[0][0], places=2)
        self.assertAlmostEqual(0.0033237, factor_returns_data.iloc[-1][0], places=2)

    def test_compute_sharpe_ratio(self) -> None:
        pricing = self.test_data_df.Close
        factor_data = alpha_factors.AverageDollarVolume(self.test_data_df, 5).for_al()
        clean_factor_data, _ = alpha_factors.prepare_alpha_lens_factor_data(factor_data.to_frame().copy(), pricing)
        factor_returns = alpha_factors.get_factor_returns(clean_factor_data)
        sharpe_ratio = alpha_factors.compute_sharpe_ratio(factor_returns, frequency='daily')['Sharpe Ratio'].values[0]
        self.assertEqual(2.1, sharpe_ratio)
        sharpe_ratio = alpha_factors.compute_sharpe_ratio(factor_returns, frequency='monthly')['Sharpe Ratio'].values[0]
        self.assertEqual(0.46, sharpe_ratio)
        sharpe_ratio = alpha_factors.compute_sharpe_ratio(factor_returns, frequency='yearly')['Sharpe Ratio'].values[0]
        self.assertEqual(0.13, sharpe_ratio)

    def test_RiskModelPCA(self) -> None:
        pricing = self.test_data_df.Close
        factor_data = alpha_factors.AverageDollarVolume(self.test_data_df, 5).for_al()
        clean_factor_data, _ = alpha_factors.prepare_alpha_lens_factor_data(factor_data.to_frame().copy(), pricing)
        factor_returns = alpha_factors.get_factor_returns(clean_factor_data)
        class_under_test = alpha_factors.RiskModelPCA(factor_returns, 1, 1)
        self.assertEqual(-1.0, class_under_test.factor_betas_.iloc[0][0])
        self.assertEqual(-1.0, class_under_test.factor_betas_.iloc[-1][0])
        self.assertAlmostEqual(5.112652940153704e-05, class_under_test.factor_cov_matrix_[0][0], 6)
        self.assertAlmostEqual(5.112652940153704e-05, class_under_test.factor_cov_matrix_[-1][0], 6)
        self.assertAlmostEqual(7.867674112695941e-36, class_under_test.idiosyncratic_var_vector_[0][0], 20)
        portfolio_variance = class_under_test.compute_portfolio_variance([.5])
        self.assertAlmostEqual(0.00357514088538959, portfolio_variance, 4)
        portfolio_variance = class_under_test.compute_portfolio_variance([1.0])
        self.assertAlmostEqual(0.00715028177077918, portfolio_variance, 4)
