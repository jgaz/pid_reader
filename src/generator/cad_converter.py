"""
Convert CAD pictures into PNGs
"""
import logging
import os
from typing import List, Dict
from generator.config import DATA_PATH, SYMBOL_SOURCE_RESOLUTIONS
from subprocess import check_output, check_call
from shutil import copyfile


logger = logging.getLogger()


def find_symbol_file(symbol_name: str) -> str:
    command = f"find {DATA_PATH}/symbol_libraries/official -name {symbol_name}.dwg".split(
        " "
    )
    path = check_output(command)
    if len(path) > 2:
        return path.strip().decode()
    raise Exception(f"File not found {command}")


def collect_dwg_file(file_path):
    destination_path = os.path.join(
        DATA_PATH, "symbol_libraries", "dwg", os.path.basename(file_path)
    )
    copyfile(file_path, destination_path)


def dxf_to_png(symbols: List[Dict[str, str]]):
    """
    'name': 'STSF014', 'family': 'Fire fighting equipment', 'description': 'WHEELED EXTINGUISHER', 'matter': 'S-Safety'
    """
    # This needs to be run in a terminal, or to have the newest librecad installed in the system
    # command = f"librecad dxf2pdf -a -k {DATA_PATH}/symbol_libraries/dxf/*.dxf"
    # check_call(command.split(" "))
    # check_call(f"mv {DATA_PATH}/symbol_libraries/dxf/*.pdf {DATA_PATH}/symbol_libraries/pdf/")

    pdf_dir = f"{DATA_PATH}/symbol_libraries/pdf/"
    png_dir = f"{DATA_PATH}/symbol_libraries/png/"

    for symbol in symbols:
        symbol_name = symbol["name"]
        pdf_name = os.path.join(pdf_dir, f"{symbol_name}.pdf")
        if os.path.isfile(pdf_name):
            for resolution in SYMBOL_SOURCE_RESOLUTIONS:
                png_file = os.path.join(png_dir, f"{resolution}", f"{symbol_name}.png")
                check_call(
                    f"convert -density {resolution} {pdf_name} -colorspace gray -threshold 99% -type bilevel -quality 100 -trim +repage {png_file}".split(
                        " "
                    )
                )
        else:
            logger.error(f"Cannot find {pdf_name}")
