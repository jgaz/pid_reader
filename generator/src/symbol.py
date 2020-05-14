import os
import string
from dataclasses import dataclass
from enum import Enum
from random import randint, choice
from typing import Tuple, Optional, Dict
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from config import PNG_SYMBOL_PATH, FONT_PATH, SYMBOL_DEBUG, DATA_PATH
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextBox:
    x: float
    y: float
    lines: int
    chars: str
    size: int
    orientation: int = 0
    size_h: int = 0
    size_w: int = 0


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


@dataclass
class CenterTextBoxConfig:
    name: str
    max_lines: int
    x: float
    y: float
    resol: int = 1


class TextBoxPosition(Enum):
    TOP = "top"
    BOTTOM = "bottom"
    RIGHT = "right"
    LEFT = "left"
    CENTER = "center"


class SymbolConfiguration:
    CSV_FILE_PATH = os.path.join(DATA_PATH, "metadata", "center_text_config.csv")
    symbol_configuration: Dict[str, CenterTextBoxConfig] = {}

    def __init__(self):
        self._read_config()

    def _read_config(self):
        symbol_config = pd.read_csv(self.CSV_FILE_PATH).to_dict("records")
        for config in symbol_config:
            self.symbol_configuration[config["name"].upper()] = CenterTextBoxConfig(
                **config
            )

    def get_config(self, symbol: GenericSymbol) -> Optional[CenterTextBoxConfig]:
        if symbol.name.upper() in self.symbol_configuration:
            return self.symbol_configuration[symbol.name.upper()]
        return None


class SymbolGenerator:
    ctbm: SymbolConfiguration
    DEFAULT_TEXT_SIZE = 15
    DEFAULT_TEXT_FONT = "font4.ttf"

    def __init__(self, ctbm: SymbolConfiguration):
        self.ctbm = ctbm

    def inject_text(
        self, symbol: GenericSymbol, diagram_image: Image.Image
    ) -> Tuple[GenericSymbol, Image.Image]:
        text_positions = [
            TextBoxPosition.TOP,
            TextBoxPosition.BOTTOM,
            TextBoxPosition.RIGHT,
            TextBoxPosition.LEFT,
            TextBoxPosition.CENTER,
        ]
        for text_position in text_positions:
            symbol.text_boxes += (self.generate_text_box(text_position, symbol),)
        draw = ImageDraw.Draw(diagram_image)

        for text_box in symbol.text_boxes:
            font = ImageFont.truetype(
                os.path.join(FONT_PATH, self.DEFAULT_TEXT_FONT), text_box.size
            )
            text_abs_coords = (
                text_box.x * symbol.size_w + symbol.x,
                text_box.y * symbol.size_h + symbol.y,
            )
            draw.text(text_abs_coords, text_box.chars, fill="black", font=font)
            text_box.size_w, text_box.size_h = draw.multiline_textsize(
                text_box.chars, font=font
            )
            text_box.x = text_abs_coords[0]
            text_box.y = text_abs_coords[1]
        return symbol, diagram_image

    def inject_symbol(self, symbol: GenericSymbol, original_image: Image):

        text_box_config = self.ctbm.get_config(symbol)
        image_quality_prefix = 225
        if text_box_config and text_box_config.resol == 2:
            image_quality_prefix = 500
        symbol_image = Image.open(
            os.path.join(PNG_SYMBOL_PATH, f"{symbol.name}_{image_quality_prefix}.png"),
            "r",
        )

        if symbol.orientation:
            symbol_image.rotate(symbol.orientation, expand=True).crop()

        original_image.paste(symbol_image, (symbol.x, symbol.y))
        symbol.size_w = symbol_image.size[0]
        symbol.size_h = symbol_image.size[1]

    def generate_text_box(
        self, type: TextBoxPosition, symbol: GenericSymbol, orientation: int = 0
    ) -> Optional[TextBox]:
        lines = randint(0, 3)
        letters = string.ascii_uppercase
        chars = ""
        size = SymbolGenerator.DEFAULT_TEXT_SIZE
        for _ in range(lines):  # Generate between 1 and 3 lines of text
            chars += "".join(choice(letters) for _ in range(5, 15)) + "\n"

        if type == TextBoxPosition.TOP:
            return TextBox(
                x=0.0,
                y=-0.3 * lines,
                lines=lines,
                chars=chars,
                size=size,
                orientation=orientation,
            )
        elif type == TextBoxPosition.BOTTOM:
            if SYMBOL_DEBUG:
                return TextBox(
                    x=0.0,
                    y=1.1,
                    lines=1,
                    chars=symbol.name,
                    size=size,
                    orientation=orientation,
                )
            return TextBox(
                x=0.0,
                y=1.1,
                lines=lines,
                chars=chars,
                size=size,
                orientation=orientation,
            )
        elif type == TextBoxPosition.RIGHT:
            return TextBox(
                x=1.1,
                y=0.2,
                lines=lines,
                chars=chars,
                size=size,
                orientation=orientation,
            )
        elif type == TextBoxPosition.LEFT:
            lines = lines % 2
            chars = "".join(choice(letters) for _ in range(3, 5))
            return TextBox(
                x=-0.3,
                y=0.1,
                lines=lines,
                chars=chars,
                size=size,
                orientation=orientation,
            )
        elif type == TextBoxPosition.CENTER:
            lines = 0
            chars = ""
            x = 0.2
            y = 0.2
            text_box_config = self.ctbm.get_config(symbol)
            if text_box_config:
                lines = text_box_config.max_lines
                x = text_box_config.x
                y = text_box_config.y

            for _ in range(lines):  # Generate between 1 and 3 lines of text
                chars += "".join(choice(letters) for _ in range(4, 7)) + "\n\n"
            return TextBox(
                x=x, y=y, lines=lines, chars=chars, size=size, orientation=orientation,
            )
        else:
            return None
