"""
Generates a set of diagrams
"""
import argparse
import os
from dataclasses import dataclass
from typing import List

from PIL import Image, ImageFont, ImageDraw
from config import DATA_PATH, DIAGRAM_PATH, FONT_PATH, PNG_SYMBOL_PATH


@dataclass
class TextBox:
    x: int
    y: int
    chars: int
    size: int


@dataclass
class GenericSymbol:
    name: str
    full_text: str
    text_boxes: List[TextBox]
    orientation: str


class SymbolFactory:
    def new_random_symbol(self, symbol_family: str) -> GenericSymbol:
        return GenericSymbol("", "", [], "")


def inject_symbol(symbol_name: str, original_image: Image) -> Image:
    img = Image.open(os.path.join(PNG_SYMBOL_PATH, f"{symbol_name}.png"), "r")
    img_w, img_h = img.size
    offset = (100, 100)
    original_image.paste(img, offset)
    return original_image


def generate_diagram(dia_type):
    img = Image.open(os.path.join(DATA_PATH, "diagram_template.png"))
    img_out_filename = os.path.join(DIAGRAM_PATH, "NewDiagram.png")
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(os.path.join(FONT_PATH, "font2.ttf"), 12)
    draw.text((0, 20), "Sample Text", fill="black", font=font)

    inject_symbol("ISCD-E001", img)
    img.save(img_out_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a set of diagrams and the input for the NNet"
    )
    parser.add_argument(
        "--number", type=int, nargs=1, help="Number of diagrams to produce", default=1
    )
    parser.add_argument(
        "--dia_type", type=str, nargs=1, help="Type of the diagram", default="random"
    )

    args = parser.parse_args()

    generate_diagram(args.dia_type)
