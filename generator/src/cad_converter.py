"""
Convert CAD pictures into PNGs
"""
import logging
import os
from typing import Union, Tuple, List
from config import DATA_PATH
from subprocess import check_output, check_call
from shutil import copyfile


logger = logging.getLogger()


def find_symbol_file(symbol_name: str) -> Union[str, bool]:
    path = check_output(
        f"find {DATA_PATH}/symbol_libraries/official -name {symbol_name}.dwg".split(" ")
    )
    if len(path) > 2:
        return path.strip().decode()
    else:
        logger.debug(f"Cannot find file for: {symbol_name}")
    return False


def collect_dwg_file(file_path):
    destination_path = os.path.join(
        DATA_PATH, "symbol_libraries", "dwg", os.path.basename(file_path)
    )
    copyfile(file_path, destination_path)


def dxf_to_png(symbols: List[Tuple[str, str, str]]):
    # This needs to be run in a terminal, or to have the newest librecad installed in the system
    # command = f"librecad dxf2pdf -a -k {DATA_PATH}/symbol_libraries/dxf/*.dxf"
    # check_call(command.split(" "))
    # check_call(f"mv {DATA_PATH}/symbol_libraries/dxf/*.pdf {DATA_PATH}/symbol_libraries/pdf/")

    pdf_dir = f"{DATA_PATH}/symbol_libraries/pdf/"
    png_dir = f"{DATA_PATH}/symbol_libraries/png/"

    for symbol in symbols:
        symbol_name = symbol[0]
        pdf_name = os.path.join(pdf_dir, f"{symbol_name}.pdf")
        if os.path.isfile(pdf_name):
            png_file = os.path.join(png_dir, f"{symbol_name}_150.png")
            check_call(
                f"convert -density 150 {pdf_name} -colorspace gray -threshold 99% -type bilevel -quality 100 {png_file}".split(
                    " "
                )
            )
            png_file = os.path.join(png_dir, f"{symbol_name}_300.png")
            check_call(
                f"convert -density 300 {pdf_name} -colorspace gray -threshold 99% -type bilevel -quality 100 {png_file}".split(
                    " "
                )
            )
        else:
            logger.error(f"Cannot find {pdf_name}")
