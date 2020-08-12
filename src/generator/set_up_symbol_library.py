"""
    Set up the symbol library used for training the network
"""
import argparse
import logging

from generator.cad_converter import find_symbol_file, collect_dwg_file, dxf_to_png
from generator.ccf_reader import parse_ccf_file
from generator.metadata import SymbolStorage


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description="Create the symbol library used for the generator"
    )
    parser.add_argument(
        "--ccf_filename", type=str, nargs=1, help="ccf filename to import"
    )
    parser.add_argument(
        "--cad_conversion", type=bool, nargs=1, help="ccf filename to import"
    )

    args = parser.parse_args()

    if args.ccf_filename:
        symbols = parse_ccf_file(args.ccf_filename[0])
        symbols_with_path = []
        for symbol in symbols:
            symbol_name = symbol[0]
            try:
                symbol_file = find_symbol_file(symbol_name)
                collect_dwg_file(symbol_file)
                file_path_split = symbol_file.split("/")
                if file_path_split[-4] == "Blocks":
                    symbols_with_path.append(symbol + (file_path_split[-3],))
            except Exception as e:
                logger.error(e)
        storage = SymbolStorage()
        storage.save(symbols_with_path)

    elif args.cad_conversion:
        storage = SymbolStorage()
        symbols_stored = storage.data.to_dict("records")
        dxf_to_png(symbols_stored)
