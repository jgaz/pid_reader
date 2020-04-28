"""
    Reads ccf files to extract symbol name and picture map
"""
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def parse_line_ccf(filename: str) -> Dict[str, Tuple]:
    symbols = {}
    with open(filename, "r") as f_in:
        for line in f_in:
            fields = line.strip().split("\t")
            if len(fields) >= 6:
                symbols[fields[0].strip()] = (fields[4].strip('"').strip(), fields[5].strip('"').strip())
    logging.debug(f"Found {len(symbols)}")
    return symbols

