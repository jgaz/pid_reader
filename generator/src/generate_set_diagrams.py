"""
Generates a set of diagrams
"""
import argparse
import os
import string
from dataclasses import dataclass
from enum import Enum
from random import randint, shuffle, choice
from typing import Tuple, Optional

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from config import DATA_PATH, DIAGRAM_PATH, PNG_SYMBOL_PATH, FONT_PATH
from metadata import SymbolStorage


@dataclass
class TextBox:
    x: float
    y: float
    lines: int
    chars: str
    size: int
    orientation: int = 0


@dataclass
class GenericSymbol:
    name: str
    x: int
    y: int
    size_h: int = 0
    size_w: int = 0
    full_text: str = ""
    text_boxes: Tuple[TextBox, ...] = ()
    orientation: int = 0


class TextBoxPosition(Enum):
    TOP = "top"
    BOTTOM = "bottom"
    RIGHT = "right"
    LEFT = "left"
    CENTER = "center"


def inject_text(
    symbol: GenericSymbol, diagram_image: Image.Image
) -> Tuple[GenericSymbol, Image.Image]:
    text_positions = [
        TextBoxPosition.TOP,
        TextBoxPosition.BOTTOM,
        TextBoxPosition.RIGHT,
    ]
    for text_position in text_positions:
        symbol.text_boxes += (generate_text_box(text_position),)
    draw = ImageDraw.Draw(diagram_image)

    for text_box in symbol.text_boxes:
        font = ImageFont.truetype(os.path.join(FONT_PATH, "font3.ttf"), text_box.size)
        draw.text(
            (
                text_box.x * symbol.size_w + symbol.x,
                text_box.y * symbol.size_h + symbol.y,
            ),
            text_box.chars,
            fill="black",
            font=font,
        )
    return symbol, diagram_image


def generate_text_box(
    type: TextBoxPosition = TextBoxPosition.TOP, orientation: int = 0
) -> Optional[TextBox]:
    lines = randint(0, 3)
    letters = string.ascii_uppercase
    chars = ""
    size = 15
    for _ in range(lines):  # Generate between 1 and 3 lines of text
        chars += "".join(choice(letters) for _ in range(5, 15)) + "\n"
    if type == TextBoxPosition.TOP:
        return TextBox(
            x=0.0,
            y=-0.2 * lines,
            lines=lines,
            chars=chars,
            size=size,
            orientation=orientation,
        )
    elif type == TextBoxPosition.BOTTOM:
        return TextBox(
            x=0.0, y=1, lines=lines, chars=chars, size=size, orientation=orientation
        )
    elif type == TextBoxPosition.RIGHT:
        return TextBox(
            x=1.0, y=0.2, lines=lines, chars=chars, size=size, orientation=orientation
        )
    else:
        return None


def inject_symbol(symbol: GenericSymbol, original_image: Image):
    symbol_image = Image.open(os.path.join(PNG_SYMBOL_PATH, f"{symbol.name}.png"), "r")
    original_image.paste(symbol_image, (symbol.x, symbol.y))
    symbol.size_w = symbol_image.size[0]
    symbol.size_h = symbol_image.size[1]


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

    for i in range(number_of_symbols):
        symbol = symbols[i % len(symbols)]
        coords = (randint(0, 5000), randint(0, 3500))
        symbol_generic = GenericSymbol(symbol.name, coords[0], coords[1])
        inject_symbol(symbol_generic, image_diagram)
        inject_text(symbol_generic, image_diagram)
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
