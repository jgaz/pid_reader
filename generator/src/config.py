import os
import logging

SRC_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SRC_PATH), "data")
TENSORFLOW_PATH = os.path.join(DATA_PATH, "tf")
DIAGRAM_PATH = os.path.join(DATA_PATH, "diagrams")
FONT_PATH = os.path.join(DATA_PATH, "fonts")
METADATA_PATH = os.path.join(DATA_PATH, "metadata")

PNG_SYMBOL_PATH = os.path.join(DATA_PATH, "symbol_libraries", "png")

SYMBOL_DEBUG = False
SYMBOL_SOURCE_RESOLUTIONS = ["100", "225", "600"]

# CONFIGURABLE PARAMETERS
DIAGRAM_SIZE = (1000, 1000)
NUMBER_OF_SYMBOLS = 4

LOGGING_LEVEL = logging.INFO

# Storage account for training data
TRAINING_STORAGE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING") or exit(
    "AZURE_STORAGE_CONNECTION_STRING needed"
)
