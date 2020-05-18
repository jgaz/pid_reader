import os

SRC_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SRC_PATH), "data")
DIAGRAM_PATH = os.path.join(DATA_PATH, "diagrams")
FONT_PATH = os.path.join(DATA_PATH, "fonts")
METADATA_PATH = os.path.join(DATA_PATH, "metadata")

PNG_SYMBOL_PATH = os.path.join(DATA_PATH, "symbol_libraries", "png")

SYMBOL_DEBUG = True
SYMBOL_SOURCE_RESOLUTIONS = ["100", "225", "600"]
