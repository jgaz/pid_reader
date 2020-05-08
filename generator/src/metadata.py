import os
import pickle
from dataclasses import dataclass
from typing import Tuple, List
import pandas as pd
from config import METADATA_PATH


import logging

logger = logging.getLogger(__name__)


@dataclass
class SymbolData:
    name: str
    family: str
    description: str


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

    def save(self, symbols: List[Tuple[str, str, str]]):
        with open(self.symbols_metadata_file, "wb") as f_out:
            pickle.dump(symbols, f_out)

    def _read(self) -> pd.DataFrame:
        if self.data is None:
            with open(self.symbols_metadata_file, "rb") as f_in:
                symbols = pickle.load(f_in)
            self.data = pd.DataFrame(
                data=symbols, columns=["name", "family", "description"]
            )
        return self.data

    def _pandas_to_symbol_data(self, df: pd.DataFrame) -> List[SymbolData]:
        symbol_list: List[SymbolData] = []
        for item in list(df.values):
            symbol_list.append(SymbolData(*item))
        return symbol_list

    def get_families(self) -> List[str]:
        return list(self.data.family.unique())

    def get_symbols_by_family(self, symbol_family: str) -> List[SymbolData]:
        df_filtered = self.data[self.data["family"] == symbol_family]
        return self._pandas_to_symbol_data(df_filtered)

    def get_dataframe(self):
        return self.data
