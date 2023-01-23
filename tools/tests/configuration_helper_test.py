import unittest
import configparser
from pathlib import Path
import tools.configuration_helper as config_helper

DEFAULT = 'DEFAULT'


class ConfigurationHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = configparser.ConfigParser()
        config.read('test_config/config.ini')
        cls.config = config

    def test_get_data_path(self):
        self.assertIsInstance(config_helper.get_data_directory(self.config[DEFAULT]), Path)

    def test_get_price_histories_path(self):
        self.assertIsInstance(config_helper.get_price_histories_path(self.config[DEFAULT]), Path)

    def test_get_alpha_factors_path(self):
        self.assertIsInstance(config_helper.get_alpha_factors_path(self.config[DEFAULT]), Path)

    def test_get_number_of_years_of_price_histories(self):
        self.assertIsInstance(config_helper.get_number_of_years_of_price_histories(self.config[DEFAULT]), str)

    def test_get_ai_alpha_path(self):
        self.assertIsInstance(config_helper.get_ai_alpha_path(self.config[DEFAULT]), Path)

    def test_get_alpha_vectors_path(self):
        self.assertIsInstance(config_helper.get_alpha_vectors_path(self.config[DEFAULT]), Path)

    def test_get_daily_betas_path(self):
        self.assertIsInstance(config_helper.get_daily_betas_path(self.config[DEFAULT]), Path)

    def test_get_ai_model_path(self):
        self.assertIsInstance(config_helper.get_ai_model_path(self.config[DEFAULT]), Path)


if __name__ == '__main__':
    unittest.main()
