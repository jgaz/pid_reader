import logging

import pandas as pd

from generator.metadata import SymbolStorage, BlockedSymbolsStorage, SymbolData

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

    def test_get_matters(self):
        ss = SymbolStorage()
        matters = ss.get_matters()
        assert len(matters) > 0
        assert matters[0] != ""

    def test_get_symbols_by_matter(self):
        ss = SymbolStorage()
        matters = ss.get_matters()
        symbols = ss.get_symbols_by_matter(matters[0])
        assert len(symbols) > 0

    def test_get_symbols_by_family(self):
        ss = SymbolStorage()
        families = ss.get_families()
        matters = ss.get_matters()
        symbols = ss.get_symbols_by_family(matter=matters[0], family=families[0])
        assert len(symbols) > 0

    def test_pandas_to_symbol_data(self):
        df = pd.DataFrame(
            [["a", "b", "c", "d"]], columns=["name", "family", "description", "matter"]
        )
        ss = SymbolStorage()
        symbol_list = ss._pandas_to_symbol_data(df)
        assert len(symbol_list) == 1
        assert symbol_list[0].name == "a"
        assert symbol_list[0].family == "b"
        assert symbol_list[0].description == "c"
        assert symbol_list[0].matter == "d"


class TestBlockedSymbolStorage:
    def test_filter_out_blocked_symbols(self):
        ss = BlockedSymbolsStorage()
        res = ss.filter_out_blocked_symbols([SymbolData("aa", "", "", "")], ["aa"])
        assert len(res) == 0
        res = ss.filter_out_blocked_symbols(
            [SymbolData("aa", "", "", ""), SymbolData("ab", "", "", "")], ["aa"]
        )
        assert len(res) == 1
