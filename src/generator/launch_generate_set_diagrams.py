"""
Generates a set of diagrams
"""
import argparse
import random
from random import shuffle
from typing import List
from PIL import Image
from generator.config import SYMBOL_DEBUG, LOGGING_LEVEL, CPU_COUNT
from generator.tensorflow_generation import generate_train_dataset
from generator.metadata import (
    SymbolStorage,
    BlockedSymbolsStorage,
    SymbolData,
)
from generator.symbol import (
    GenericSymbol,
    SymbolGenerator,
    SymbolConfiguration,
    SymbolPositioner,
)
import logging
import multiprocessing

from generator.training_storage import (
    DiagramStorage,
    DiagramSymbolsStorage,
    TrainingDatasetLabelDictionaryStorage,
)

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


def get_valid_symbols(
    symbol_storage: SymbolStorage, diagram_matters: List[str]
) -> List[SymbolData]:
    symbols: List[SymbolData] = []

    for matter in diagram_matters:
        symbols.extend(symbol_storage.get_symbols_by_matter(matter))

    blocked_symbol_st = BlockedSymbolsStorage()
    symbols = blocked_symbol_st.filter_out_blocked_symbols(
        symbols, blocked_symbol_st.blocked_symbols
    )
    return symbols


def generate_diagram(params):
    dss, diagram_size, symbols_per_diagram, symbols = params
    possible_orientation = [90, 0]

    image_diagram = Image.new("LA", diagram_size, 255)

    shuffle(symbols)
    diagram_symbols = []
    ctbm = SymbolConfiguration()
    symbol_generator = SymbolGenerator(ctbm=ctbm, diagram_size=diagram_size)
    positions = SymbolPositioner.get_symbol_position(symbols_per_diagram, diagram_size)

    for i in range(symbols_per_diagram):
        symbol = symbols[i % len(symbols)]
        coords = positions[i]
        orientation = random.choice(possible_orientation)
        symbol_generic = GenericSymbol(
            symbol.name, coords[0], coords[1], orientation=orientation
        )
        symbol_generator.inject_symbol(symbol_generic, image_diagram)
        if SYMBOL_DEBUG:
            symbol_generator.draw_boxes(symbol_generic, image_diagram)
        diagram_symbols.append(symbol_generic)

    diagram_storage = DiagramStorage()
    diagram_storage.store_image(dss, image_diagram, diagram_symbols)
    return diagram_symbols


def generate_set_diagrams(
    dss, diagram_size, symbols_per_diagram, valid_symbols, number_diagrams
):
    DiagramStorage.clear()
    params = [
        (dss, diagram_size, symbols_per_diagram, valid_symbols)
        for _ in range(number_diagrams)
    ]
    pool = multiprocessing.Pool(CPU_COUNT)
    pool.map(generate_diagram, params)
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a set of diagrams and the input for the NNet"
    )

    parser.add_argument(
        "--number_diagrams",
        type=int,
        nargs=1,
        help="Number of diagrams to produce",
        default=[100],
    )

    parser.add_argument(
        "--symbols_per_diagram",
        type=int,
        nargs=1,
        help="Symbols per diagram",
        default=[10],
    )

    parser.add_argument(
        "--diagram_matter",
        type=str,
        nargs="*",
        help="""Matters of the diagram, at least two""",
        choices=[
            "P-Process",
            "L-Piping",
            "J-Instrument",
            "H-HVAC",
            "T-telecom",
            "N-Structural",
            "R-Mechanical",
            "E-Electro",
            "S-Safety",
        ],
        default=None,
        required=True,
    )

    parser.add_argument(
        "--diagram_size",
        type=int,
        nargs=2,
        help="Diagram size: 500 500",
        default=[1000, 1000],
    )
    args = parser.parse_args()

    symbol_storage = SymbolStorage()
    dss = DiagramSymbolsStorage()

    number_diagrams = int(args.number_diagrams[0])
    symbols_per_diagram = int(args.symbols_per_diagram[0])
    diagram_size = args.diagram_size

    if args.diagram_matter:
        if type(args.diagram_matter) == list:
            diagram_matters = args.diagram_matter
        else:
            diagram_matters = [args.diagram_matter]
    else:
        matters = symbol_storage.get_matters()
        random.choices(matters, k=2)

    valid_symbols = get_valid_symbols(symbol_storage, diagram_matters)

    # Generate the diagrams in multiprocess
    generate_set_diagrams(
        dss, diagram_size, symbols_per_diagram, valid_symbols, number_diagrams
    )

    TrainingDatasetLabelDictionaryStorage.save(valid_symbols)

    # Generate tensorflow datasets and push it to Azure blob
    model_id = generate_train_dataset(diagram_matters)
    print(f"Dataset id generated: {model_id}")
