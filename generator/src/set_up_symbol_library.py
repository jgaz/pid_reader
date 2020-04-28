"""
    Set up the symbol library used for training the network
"""
import argparse

from cad_converter import find_symbol_file, collect_dwg_file, dxf_to_png
from ccf_reader import parse_line_ccf
from metadata import Storage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the symbol library used for the generator')
    parser.add_argument('--ccf_filename', type=str, nargs=1, help='ccf filename to import')
    parser.add_argument('--cad_conversion', type=bool, nargs=1, help='ccf filename to import')

    args = parser.parse_args()

    if args.ccf_filename:
        symbols = parse_line_ccf(args.ccf_filename[0])
        for symbol_name, symbol_info in symbols.items():
            symbol_file = find_symbol_file(symbol_name)
            if symbol_file:
                collect_dwg_file(symbol_file)
        storage = Storage()
        storage.save_symbols(symbols)

    elif args.cad_conversion:
        dxf_to_png()