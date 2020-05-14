import os
import pickle
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List
from config import METADATA_PATH


import logging

logger = logging.getLogger(__name__)


@dataclass
class SymbolData:
    name: str
    family: str
    description: str
    matter: str


class SymbolStorage:
    symbols_metadata_file = os.path.join(METADATA_PATH, "symbols.pickle")
    data: pd.DataFrame = None

    def __init__(self):
        try:
            self._read()
        except Exception as e:
            logger.error(
                "Cannot read the source file for symbols, check that you have generated it"
            )
            raise e

    def save(self, symbols: List[Tuple[str, ...]]):
        with open(self.symbols_metadata_file, "wb") as f_out:
            pickle.dump(symbols, f_out)

    def _read(self) -> pd.DataFrame:
        if self.data is None:
            with open(self.symbols_metadata_file, "rb") as f_in:
                symbols = pickle.load(f_in)
            self.data = pd.DataFrame(
                data=symbols, columns=["name", "family", "description", "matter"]
            )
        return self.data

    def _pandas_to_symbol_data(self, df: pd.DataFrame) -> List[SymbolData]:
        symbol_list: List[SymbolData] = []
        for item in list(df.values):
            symbol_list.append(SymbolData(*item))
        return symbol_list

    def get_families(self) -> List[str]:
        return list(self.data.family.unique())

    def get_matters(self) -> List[str]:
        return list(self.data.matter.unique())

    def get_symbols_by_family(self, matter: str, family: str) -> List[SymbolData]:
        df_filtered = self.data.loc[
            (self.data.matter == matter) & (self.data.family == family)
        ]
        return self._pandas_to_symbol_data(df_filtered)

    def get_symbols_by_matter(self, matter: str) -> List[SymbolData]:
        df_filtered = self.data.loc[(self.data.matter == matter)]
        return self._pandas_to_symbol_data(df_filtered)

    def get_dataframe(self):
        return self.data


class BlockedSymbolsStorage:
    blocked_symbols: List[str] = []
    BLOCKED_SYMBOLS_METADATA_FILE = os.path.join(METADATA_PATH, "symbols_blocked.csv")

    def __init__(self):
        self._read()

    def _read(self) -> List[str]:
        if not self.blocked_symbols:
            df = pd.read_csv(self.BLOCKED_SYMBOLS_METADATA_FILE)
            self.blocked_symbols = [x.upper() for x in df.name.values]

        return self.blocked_symbols

    def filter_out_blocked_symbols(
        self, symbols: List[SymbolData], blocked_symbols: List[str]
    ):
        set_blocked_symbols = set([x.upper() for x in blocked_symbols])
        return [s for s in symbols if s.name.upper() not in set_blocked_symbols]
