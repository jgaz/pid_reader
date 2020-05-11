"""
Generates a set of diagrams
"""
import argparse
import os
from random import randint, shuffle
from PIL import Image
from config import DATA_PATH, DIAGRAM_PATH
from metadata import SymbolStorage
from symbol import GenericSymbol, SymbolGenerator, CenterTextBoxManager


def generate_diagram(diagram_matter: str):
    number_of_symbols = randint(50, 100)
    image_diagram = Image.open(os.path.join(DATA_PATH, "diagram_template.png"))
    img_out_filename = os.path.join(DIAGRAM_PATH, "NewDiagram.png")

    symbol_st = SymbolStorage()
    if diagram_matter != "random":
        symbols = symbol_st.get_symbols_by_matter(diagram_matter)
    else:
        symbols = symbol_st.get_symbols_by_matter(symbol_st.get_matters()[0])
    shuffle(symbols)

    diagram_symbols = []
    ctbm = CenterTextBoxManager()
    symbol_generator = SymbolGenerator(ctbm=ctbm)
    for i in range(number_of_symbols):
        symbol = symbols[i % len(symbols)]
        coords = (randint(0, 5000), randint(0, 3500))
        symbol_generic = GenericSymbol(symbol.name, coords[0], coords[1])
        symbol_generator.inject_symbol(symbol_generic, image_diagram)
        symbol_generator.inject_text(symbol_generic, image_diagram)
        diagram_symbols.append(symbol)
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
        nargs=1,
        help="Matter of the diagram",
        default="random",
    )

    args = parser.parse_args()

    symbols_used = generate_diagram(args.diagram_matter[0])
    print(symbols_used)
