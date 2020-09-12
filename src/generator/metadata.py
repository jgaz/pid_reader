import hashlib
import json
import os
import pickle
import shutil

import PIL
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, TypedDict, Any, Dict
from generator.config import METADATA_PATH, DIAGRAM_PATH, DIAGRAM_CLASSES_FILE
import logging
from generator.symbol import GenericSymbol
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


class DiagramSymbolsStorage:
    """
    Storage of the symbol metadata: position, type, etc...
    """

    PATH = DIAGRAM_PATH

    def _get_path(self, hash: str):
        return os.path.join(DiagramSymbolsStorage.PATH, f"Diagram_{hash}.pickle")

    def save(self, hash: str, symbols: List[GenericSymbol]):
        pickle.dump(symbols, open(self._get_path(hash), "wb"))

    def load(self, hash: str = None, filename: str = None):
        if filename:
            return pickle.load(open(filename, "rb"))
        elif hash:
            return pickle.load(open(self._get_path(hash), "rb"))


class DiagramStorage:
    def store_image(self, dss: DiagramSymbolsStorage, image_diagram, diagram_symbols):
        image: PIL.Image = image_diagram.convert("1")
        hash = hashlib.md5(image.tobytes()).hexdigest()
        image.save(os.path.join(DIAGRAM_PATH, f"Diagram_{hash}.png"))
        # Store symbols too
        dss.save(hash, diagram_symbols)

    @staticmethod
    def clear():
        shutil.rmtree(DIAGRAM_PATH)
        os.makedirs(DIAGRAM_PATH)


class TrainingDatasetLabelDictionaryStorage:
    @staticmethod
    def save(valid_symbols: List[SymbolData]):
        logger.info("Saving symbols dictionary")
        valid_symbols_dict = {}
        for i, symbol in enumerate(valid_symbols):
            valid_symbols_dict[symbol.name] = i + 1
        file_path = os.path.join(DIAGRAM_PATH, DIAGRAM_CLASSES_FILE)
        json.dump(valid_symbols_dict, open(file_path, "w"))

    @staticmethod
    def get(data_path: str) -> Dict[str, int]:
        """
        Get the object names and Ids
        :param data_path: path of the class file
        :return:
        """
        classes_filename = os.path.join(data_path, DIAGRAM_CLASSES_FILE)
        dictionary = json.load(open(classes_filename, "r"))
        return dictionary
