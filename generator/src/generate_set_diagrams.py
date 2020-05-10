"""
Generates a set of diagrams
"""
import argparse
import os
from dataclasses import dataclass
from random import randint, shuffle
from typing import Tuple

from PIL import Image
from config import DATA_PATH, DIAGRAM_PATH, PNG_SYMBOL_PATH
from metadata import SymbolStorage


@dataclass
class TextBox:
    x: int
    y: int
    chars: int
    size: int


@dataclass
class GenericSymbol:
    name: str
    x: int
    y: int
    size_h: int = 0
    size_w: int = 0
    full_text: str = ""
    text_boxes: Tuple[TextBox, ...] = ()
    orientation: str = ""


def inject_symbol(symbol: GenericSymbol, original_image: Image):
    symbol_image = Image.open(os.path.join(PNG_SYMBOL_PATH, f"{symbol.name}.png"), "r")
    original_image.paste(symbol_image, (symbol.x, symbol.y))
    symbol.size_w = symbol_image.size[0]
    symbol.size_h = symbol_image.size[1]


def generate_diagram(diagram_matter: str):
    number_of_symbols = randint(50, 100)
    img = Image.open(os.path.join(DATA_PATH, "diagram_template.png"))
    img_out_filename = os.path.join(DIAGRAM_PATH, "NewDiagram.png")

    symbol_st = SymbolStorage()
    if diagram_matter != "random":
        symbols = symbol_st.get_symbols_by_matter(diagram_matter)
    else:
        symbols = symbol_st.get_symbols_by_matter(symbol_st.get_matters()[0])
    shuffle(symbols)

    # draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(os.path.join(FONT_PATH, "font2.ttf"), 12)
    # draw.text((0, 20), "Sample Text", fill="black", font=font)

    diagram_symbols = []

    for i in range(number_of_symbols):
        symbol = symbols[i % len(symbols)]
        coords = (randint(0, 5000), randint(0, 3500))
        symbol_generic = GenericSymbol(symbol.name, coords[0], coords[1])
        inject_symbol(symbol_generic, img)
        diagram_symbols.append(symbol)
        img.save(img_out_filename)
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
