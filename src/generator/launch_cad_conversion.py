"""
    Convert registered symbols in the library with
    CAD files into PNGs
"""
import logging

from generator.cad_converter import dxf_to_png
from generator.metadata import SymbolStorage


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    storage = SymbolStorage()
    symbols_stored = storage.data.to_dict("records")
    dxf_to_png(symbols_stored)
