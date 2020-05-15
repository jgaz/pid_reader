"""
Generates a set of diagrams
"""
import argparse
import os
from pprint import pprint
from random import randint, shuffle
from typing import List, Optional

from PIL import Image
from config import DIAGRAM_PATH
from metadata import SymbolStorage, BlockedSymbolsStorage
from symbol import GenericSymbol, SymbolGenerator, SymbolConfiguration


def generate_diagram(diagram_matters: Optional[List[str]]):
    number_of_symbols = randint(100, 200)
    possible_orientation = [90]
    image_diagram = Image.new("LA", (5000, 3500), 255)
    img_out_filename = os.path.join(DIAGRAM_PATH, "NewDiagram.png")

    symbol_st = SymbolStorage()
    if diagram_matters:
        symbols = []
        for matter in diagram_matters:
            symbols.extend(symbol_st.get_symbols_by_matter(matter))
    else:
        symbols = symbol_st.get_symbols_by_matter(symbol_st.get_matters()[0])

    blocked_symbol_st = BlockedSymbolsStorage()
    symbols = blocked_symbol_st.filter_out_blocked_symbols(
        symbols, blocked_symbol_st.blocked_symbols
    )

    shuffle(symbols)
    diagram_symbols = []
    ctbm = SymbolConfiguration()
    symbol_generator = SymbolGenerator(ctbm=ctbm)
    for i in range(number_of_symbols):
        symbol = symbols[i % len(symbols)]
        coords = (randint(0, 5000), randint(0, 3500))
        orientation = possible_orientation[0] if i % 4 == 0 else 0
        symbol_generic = GenericSymbol(
            symbol.name, coords[0], coords[1], orientation=orientation
        )
        symbol_generator.inject_symbol(symbol_generic, image_diagram)
        symbol_generator.draw_boxes(symbol_generic, image_diagram)
        diagram_symbols.append(symbol_generic)
    image_diagram = image_diagram.convert("1")
    image_diagram.save(img_out_filename)
    return diagram_symbols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a set of diagrams and the input for the NNet"
    )
    parser.add_argument(
        "--number", type=int, nargs=1, help="Number of diagrams to produce", default=1
    )
    parser.add_argument(
        "--diagram_matter",
        type=str,
        nargs="*",
        help="Matter of the diagram",
        default=None,
    )

    args = parser.parse_args()
    symbols_used = generate_diagram(args.diagram_matter)
    pprint(symbols_used)
