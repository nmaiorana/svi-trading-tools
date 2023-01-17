from configparser import SectionProxy
from pathlib import Path


def get_alpha_factors_path(configuration: SectionProxy):
    return Path(configuration["DataDirectory"] + '/' + configuration["AlphaFactorsFileName"])
