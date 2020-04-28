import os
import pickle
from typing import Dict, Tuple

from config import METADATA_PATH


class Storage:
    symbols_metadata_file = os.path.join(METADATA_PATH, "symbols.pickle")

    def save_symbols(self, symbols: Dict[str, Tuple]):
        with open(self.symbols_metadata_file, "wb") as f_out:
            pickle.dump(symbols, f_out)

    def read_symbols(self) -> Dict[str, Tuple]:
        with open(self.symbols_metadata_file, "rb") as f_in:
            return pickle.load(f_in)
