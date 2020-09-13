import os
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, TypedDict, Any
from generator.config import METADATA_PATH, PNG_SYMBOL_PATH
import logging
import csv

logger = logging.getLogger(__name__)


@dataclass
class SymbolData:
    name: str
    family: str
    description: str
    matter: str


class JsonTrainingObject(TypedDict):
    images: List[str]
    type: str
    annotations: List[Any]
    categories: List[Any]


class SymbolStorage:
    symbols_metadata_file = os.path.join(METADATA_PATH, "symbols.csv")
    columns = ["name", "family", "description", "matter"]
    data: pd.DataFrame = None

    def __init__(self):
        try:
            self._read()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            logger.error(
                "Cannot read the source file for symbols, generating a new one?"
            )
        except Exception as e:
            raise e

    def save(self, symbols: List[Tuple[str, ...]]):
        with open(self.symbols_metadata_file, "w") as f_out:
            csv_writer = csv.writer(f_out)
            csv_writer.writerow(self.columns)
            csv_writer.writerows(symbols)
        logger.info("Generated new symbol library")

    def _read(self) -> pd.DataFrame:
        if self.data is None:
            self.data = pd.read_csv(self.symbols_metadata_file, header=0)
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

    def get_html_visualization(self):
        formatters = {
            "name": lambda x: f"<image src='{PNG_SYMBOL_PATH}/225/{x}.png'><br>{x}"
        }
        return self.data.to_html(formatters=formatters, escape=False)


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
