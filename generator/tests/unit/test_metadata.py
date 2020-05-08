import logging

import pandas as pd

from metadata import SymbolStorage

logger = logging.getLogger(__name__)


class TestSymbolStorage:
    def test_get_dataframe(self):
        ss = SymbolStorage()
        df = ss.get_dataframe()
        assert len(df.name) > 0

    def test_get_families(self):
        ss = SymbolStorage()
        families = ss.get_families()
        assert len(families) > 0
        assert families[0] != ""

    def test_get_symbols_by_family(self):
        ss = SymbolStorage()
        families = ss.get_families()
        symbols = ss.get_symbols_by_family(families[0])
        assert len(symbols) > 0

    def test_pandas_to_symbol_data(self):
        df = pd.DataFrame([["a", "b", "c"]], columns=["name", "family", "description"])
        ss = SymbolStorage()
        symbol_list = ss._pandas_to_symbol_data(df)
        assert len(symbol_list) == 1
        assert symbol_list[0].name == "a"
        assert symbol_list[0].family == "b"
        assert symbol_list[0].description == "c"
