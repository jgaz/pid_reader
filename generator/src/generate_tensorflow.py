"""

"""
from pathlib import Path
from pprint import pprint

from config import DIAGRAM_PATH

# List all files in Data
from metadata import DiagramSymbolsStorage

diagram_path = Path(DIAGRAM_PATH)
ss = DiagramSymbolsStorage()
for diagrapm_data_file in diagram_path.glob("*.pickle"):

    file_hash = diagrapm_data_file.name.split(".")[0].split("_")[1]
    metadata = ss.load(file_hash)
    """ GenericSymbol(
    name='STPS009', x=205, y=219, size_h=57, size_w=106, full_text='',
    text_boxes=(TextBox(x=205.0, y=198.0, lines=1, chars='FMCPDSLIWP', size=15, orientation=0, size_h=17, size_w=106),
                TextBox(x=205.0, y=281.7, lines=1, chars='KSMCUBDYBG', size=15, orientation=0, size_h=17, size_w=110),
                TextBox(x=175.0, y=224.7, lines=1, chars='VH', size=15, orientation=0, size_h=17, size_w=23)),
                orientation=0, resolution=1)"""


# Get the categories and Ids

# Image by image, read it

# Get the pickle information

# Create the TF Record
