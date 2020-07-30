import os
import logging

SRC_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(SRC_PATH)), "data_generator")
TENSORFLOW_PATH = os.path.join(DATA_PATH, "tf")
DIAGRAM_PATH = os.path.join(DATA_PATH, "diagrams")
FONT_PATH = os.path.join(DATA_PATH, "fonts")
METADATA_PATH = os.path.join(DATA_PATH, "metadata")
PNG_SYMBOL_PATH = os.path.join(DATA_PATH, "symbol_libraries", "png")

GENERATOR_METADATA_FILE = "training_metadata.yaml"
GENERATOR_LABEL_FILE = "label_map.pbtxt"

SYMBOL_DEBUG = False
SYMBOL_SOURCE_RESOLUTIONS = ["100", "225", "600"]

CPU_COUNT = len(os.sched_getaffinity(0)) // 2  # Use half of the CPUs

# CONFIGURABLE PARAMETERS
LOGGING_LEVEL = logging.INFO
