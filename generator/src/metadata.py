import os
import pickle
from dataclasses import dataclass
from typing import Tuple, List
import pandas as pd
from config import METADATA_PATH


@dataclass
class SymbolData:
    name: str
    family: str
    description: str


class SymbolStorage:
    symbols_metadata_file = os.path.join(METADATA_PATH, "symbols.pickle")
    data: pd.DataFrame = None

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

    def get_families(self) -> List[str]:
        families: List[str] = []
        self._read()
        return families

    def get_symbols_by_family(self, symbol_family: str) -> List[SymbolData]:
        symbols: List[SymbolData] = []
        self._read()
        return symbols

    def get_dataframe(self):
        self._read()
        return self.data
