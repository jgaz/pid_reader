import os
import string
from dataclasses import dataclass
from enum import Enum
from math import floor
from random import randint, choice
from typing import Tuple, Optional, Dict, List
import pandas as pd
from PIL import Image, ImageOps
from PIL import ImageDraw
from PIL import ImageFont
from generator.config import (
    PNG_SYMBOL_PATH,
    FONT_PATH,
    SYMBOL_DEBUG,
    DATA_PATH,
    SYMBOL_SOURCE_RESOLUTIONS,
)
import logging
from fa2 import ForceAtlas2
import numpy as np

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
    resolution: int = 1


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
    ASSEMBLY_IMAGE_SIZE = (400, 400)
    ASSEMBLY_IMAGE_OFFSET = (ASSEMBLY_IMAGE_SIZE[0] // 10, ASSEMBLY_IMAGE_SIZE[1] // 10)

    def __init__(self, ctbm: SymbolConfiguration, diagram_size: Tuple[int, int]):
        self.ctbm = ctbm
        self.DIAGRAM_SIZE = diagram_size

    def inject_text(
        self, symbol: GenericSymbol, diagram_image: Image.Image, offset: Tuple[int, int]
    ) -> Tuple[GenericSymbol, Image.Image]:
        text_positions = [
            TextBoxPosition.TOP,
            TextBoxPosition.BOTTOM,
            TextBoxPosition.RIGHT,
            TextBoxPosition.LEFT,
            TextBoxPosition.CENTER,
        ]

        for text_position in text_positions:

            text_box = self.generate_text_box(text_position, symbol)
            if text_box:
                symbol.text_boxes += (text_box,)
        draw = ImageDraw.Draw(diagram_image)

        for text_box in symbol.text_boxes:
            font = ImageFont.truetype(
                os.path.join(FONT_PATH, self.DEFAULT_TEXT_FONT), text_box.size
            )
            text_coords = (
                floor(text_box.x * symbol.size_w + offset[0]),
                floor(text_box.y * symbol.size_h + offset[1]),
            )
            # Bug: Fedora has a problem in the library, it throws core dumps, Debian does fine
            draw.text(text_coords, text_box.chars, fill=0, font=font)

            text_box.size_w, text_box.size_h = draw.multiline_textsize(
                text_box.chars, font=font
            )
            # Set the relative coords for the text box
            text_box.x, text_box.y = text_coords

        return symbol, diagram_image

    def inject_symbol(self, symbol: GenericSymbol, original_image: Image):
        # Fetch the image in appropriate format
        text_box_config = self.ctbm.get_config(symbol)
        resolution = text_box_config.resol if text_box_config else 1
        image_quality_prefix = SYMBOL_SOURCE_RESOLUTIONS[resolution]
        symbol_image = Image.open(
            os.path.join(
                PNG_SYMBOL_PATH, f"{image_quality_prefix}", f"{symbol.name}.png"
            ),
            "r",
        )
        symbol.resolution = resolution

        # Put symbol and text boxes
        offset = self.ASSEMBLY_IMAGE_OFFSET
        assemble_image = Image.new("L", self.ASSEMBLY_IMAGE_SIZE, 255)
        assemble_image.paste(symbol_image, offset)
        symbol.size_w, symbol.size_h = symbol_image.size

        # Move symbol if it is going to be plotted outside the visible region
        symbol = self.reposition_inside_visible(symbol)

        self.inject_text(symbol, assemble_image, offset)
        if symbol.orientation > 0:
            assemble_image = assemble_image.rotate(symbol.orientation, expand=True)

        # Paste the generated symbol in the diagram
        inverted_image = ImageOps.invert(assemble_image)
        assemble_image.putalpha(inverted_image)

        original_image.paste(assemble_image, (symbol.x, symbol.y), mask=inverted_image)
        # Recalculate positioning after the paste [and rotation]
        self.recalculate_positions(symbol, offset)

    def reposition_inside_visible(self, symbol: GenericSymbol) -> GenericSymbol:
        if (
            symbol.orientation > 0
        ):  # This is problematic as the full image is rotated, adding a lot of padding
            extra_padding = self.ASSEMBLY_IMAGE_SIZE[1] - self.ASSEMBLY_IMAGE_OFFSET[1]
            if symbol.y + extra_padding > self.DIAGRAM_SIZE[1] - symbol.size_w:
                symbol.y -= extra_padding

            if (
                symbol.x + symbol.size_h + self.ASSEMBLY_IMAGE_OFFSET[0]
                > self.DIAGRAM_SIZE[0]
            ):
                symbol.x = (
                    self.DIAGRAM_SIZE[0] - symbol.size_h - self.ASSEMBLY_IMAGE_OFFSET[0]
                )
        else:
            if (
                symbol.x + symbol.size_w + self.ASSEMBLY_IMAGE_OFFSET[0]
                > self.DIAGRAM_SIZE[0]
            ):
                symbol.x = (
                    self.DIAGRAM_SIZE[0] - symbol.size_w - self.ASSEMBLY_IMAGE_OFFSET[0]
                )
            if (
                symbol.y + symbol.size_h + self.ASSEMBLY_IMAGE_OFFSET[1]
                > self.DIAGRAM_SIZE[1]
            ):
                symbol.y = (
                    self.DIAGRAM_SIZE[1] - symbol.size_h - self.ASSEMBLY_IMAGE_OFFSET[1]
                )
        return symbol

    def recalculate_positions(self, symbol: GenericSymbol, offset: Tuple[int, int]):
        if symbol.orientation == 90:
            old_symbol = (int(symbol.x), int(symbol.y))
            symbol.y += -offset[0] + self.ASSEMBLY_IMAGE_SIZE[0] - symbol.size_w
            symbol.x += offset[1]
            symbol.size_h, symbol.size_w = (symbol.size_w, symbol.size_h)

            for text_box in symbol.text_boxes:
                old_y = int(text_box.y)
                old_x = int(text_box.x)
                text_box.y = (
                    old_symbol[1]
                    - old_x
                    + self.ASSEMBLY_IMAGE_SIZE[0]
                    - text_box.size_w
                )
                text_box.x = symbol.x + old_y - self.ASSEMBLY_IMAGE_OFFSET[0]
                text_box.size_h, text_box.size_w = (text_box.size_w, text_box.size_h)
        else:
            for text_box in symbol.text_boxes:
                text_box.x += symbol.x
                text_box.y += symbol.y
            symbol.x += offset[0]
            symbol.y += offset[1]

        return symbol

    def generate_text_box(
        self, type: TextBoxPosition, symbol: GenericSymbol
    ) -> Optional[TextBox]:
        lines = randint(0, 3)
        letters = string.ascii_uppercase
        chars = ""
        size = SymbolGenerator.DEFAULT_TEXT_SIZE
        if symbol.resolution == 0:
            size = 10

        for _ in range(lines):  # Generate between 1 and 3 lines of text
            chars += "".join(choice(letters) for _ in range(5, 15)) + "\n"
        chars = chars[:-1]

        if type == TextBoxPosition.TOP:
            y = (lines * self.DEFAULT_TEXT_SIZE) * 1.4 / symbol.size_h
            if lines > 0:
                return TextBox(
                    x=0.0,
                    y=-y,
                    lines=lines,
                    chars=chars,
                    size=size,
                    orientation=symbol.orientation,
                )
        elif type == TextBoxPosition.BOTTOM:
            if lines > 0:
                if SYMBOL_DEBUG:
                    return TextBox(
                        x=0.0,
                        y=1.1,
                        lines=1,
                        chars=symbol.name,
                        size=size,
                        orientation=symbol.orientation,
                    )
                return TextBox(
                    x=0.0,
                    y=1.1,
                    lines=lines,
                    chars=chars,
                    size=size,
                    orientation=symbol.orientation,
                )
        elif type == TextBoxPosition.RIGHT:
            if lines > 0:
                return TextBox(
                    x=1.01,
                    y=0.2,
                    lines=lines,
                    chars=chars,
                    size=size,
                    orientation=symbol.orientation,
                )
        elif type == TextBoxPosition.LEFT:
            if lines > 0:
                lines = lines % 2
                chars = "".join(choice(letters) for _ in range(3, 5))
                x = len(chars) * self.DEFAULT_TEXT_SIZE / symbol.size_w
                return TextBox(
                    x=-x,
                    y=0.1,
                    lines=lines,
                    chars=chars,
                    size=size,
                    orientation=symbol.orientation,
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
            if lines > 0:
                for _ in range(lines):  # Generate between 1 and 3 lines of text
                    chars += "".join(choice(letters) for _ in range(4, 7)) + "\n\n"
                chars = chars[:-2]
                return TextBox(
                    x=x,
                    y=y,
                    lines=lines,
                    chars=chars,
                    size=size,
                    orientation=symbol.orientation,
                )

        return None

    def draw_boxes(self, symbol: GenericSymbol, original_image: Image):
        # Test out positioning reported
        draw = ImageDraw.Draw(original_image)
        box = (
            (symbol.x, symbol.y),
            (symbol.x + symbol.size_w, symbol.y),
            (symbol.x + symbol.size_w, symbol.y + symbol.size_h),
            (symbol.x, symbol.y + symbol.size_h),
        )
        draw.polygon(box, outline="blue")

        for text_box in symbol.text_boxes:
            box2 = (
                (text_box.x, text_box.y),
                (text_box.x + text_box.size_w, text_box.y),
                (text_box.x + text_box.size_w, text_box.y + text_box.size_h),
                (text_box.x, text_box.y + text_box.size_h),
            )
            draw.polygon(box2, outline="blue")


class SymbolPositioner:
    @staticmethod
    def get_symbol_position(
        number_of_symbols: int, diagram_size: Tuple[int, int]
    ) -> List[Tuple[int, int]]:

        if number_of_symbols == 1:
            positions = [
                (randint(0, diagram_size[0]), randint(0, diagram_size[1])),
            ]
        else:
            forceatlas2 = ForceAtlas2(
                # Behavior alternatives
                outboundAttractionDistribution=True,  # Dissuade hubs
                linLogMode=False,  # NOT IMPLEMENTED
                adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                edgeWeightInfluence=1.0,
                # Performance
                jitterTolerance=0.1,  # Tolerance
                barnesHutOptimize=True,
                barnesHutTheta=1.2,
                multiThreaded=False,  # NOT IMPLEMENTED
                # Tuning
                scalingRatio=2.0,
                strongGravityMode=False,
                gravity=0.5,
                # Log
                verbose=False,
            )
            G = np.identity(number_of_symbols)
            positions = forceatlas2.forceatlas2(G, pos=None, iterations=50)

            x, y = zip(*positions)
            x_range = max(x) - min(x) or 1
            y_range = max(y) - min(y) or 1
            dx, dy = (diagram_size[0], diagram_size[1])
            x_ratio = dx / x_range
            y_ratio = dy / y_range
            xnp = (np.array(x) + np.abs(min(x))) * x_ratio
            ynp = (np.array(y) + np.abs(min(y))) * y_ratio
            positions = list(zip(xnp.astype(int), ynp.astype(int)))
        return positions
