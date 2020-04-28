"""
Convert CAD pictures into PNGs
"""
import logging
import os
from typing import Union
from config import DATA_PATH
from subprocess import check_output, check_call
from shutil import copyfile


logger = logging.getLogger()


def find_symbol_file(symbol_name: str) -> Union[str, bool]:
    path = check_output(f"find {DATA_PATH}/symbol_libraries/official -name {symbol_name}.dwg".split(" "))
    if len(path) > 2:
        return path.strip().decode()
    else:
        logger.debug(f"Cannot find file for: {symbol_name}")


def collect_dwg_file(file_path):
    destination_path = os.path.join(DATA_PATH, "symbol_libraries", "dwg", os.path.basename(file_path))
    copyfile(file_path, destination_path)


def dxf_to_png():
    # This needs to be run in a terminal, or to have the newest librecad installed in the system
    # command = f"librecad dxf2pdf -a -k {DATA_PATH}/symbol_libraries/dxf/*.dxf"
    # check_call(command.split(" "))
    # check_call(f"mv {DATA_PATH}/symbol_libraries/dxf/*.pdf {DATA_PATH}/symbol_libraries/pdf/")

    for pdf_file in os.scandir(f"{DATA_PATH}/symbol_libraries/pdf/"):
        if pdf_file.is_file():
            png_file = f"{DATA_PATH}/symbol_libraries/png/{pdf_file.name[:-3]}.png"

            check_call(f"gs -sDEVICE=png16m -dNOPAUSE -dBATCH -dSAFER -r300 -sOutputFile={png_file} {os.path.join(pdf_file.path)}".split(" "))

