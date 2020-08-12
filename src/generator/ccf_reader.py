"""
    Reads ccf files to extract symbol name and picture map
"""
import logging
import re
from typing import Tuple, List

logger = logging.getLogger(__name__)


def parse_ccf_file(filename: str) -> List[Tuple[str, ...]]:
    symbols = {}
    with open(filename, "r") as f_in:
        for line in f_in:
            fields = line.strip().split("\t")
            if len(fields) >= 6:
                symbols[fields[0].strip()] = (
                    fields[0].strip(),
                    fields[4].strip('"').strip(),
                    fields[5].strip('"').strip(),
                )
    filtered_symbols = remove_bad_ccf_symbol_family(list(symbols.values()))
    logger.info(f"Found {len(filtered_symbols)}")
    return filtered_symbols


def remove_bad_ccf_symbol_family(
    symbol_list: List[Tuple[str, ...]]
) -> List[Tuple[str, ...]]:
    bad_symbol_family = [
        re.compile(r"^EU-.*"),
        re.compile(r"^UK-.*"),
        re.compile(r"^US-.*"),
    ]

    for bad_symbol in bad_symbol_family:
        symbol_list = [x for x in symbol_list if bad_symbol.match(x[1]) is None]

    return symbol_list
