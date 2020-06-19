"""

"""
import argparse
import os
from pathlib import Path
from typing import List

from config import DIAGRAM_PATH, LOGGING_LEVEL
import logging
from metadata import DiagramSymbolsStorage, TensorflowStorage, SymbolStorage
import multiprocessing
import tensorflow.compat.v1 as tf

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


def get_categories_map(categories: List[str]):
    # Get the object names and Ids
    ss = SymbolStorage()
    df = ss.get_dataframe()
    df = df[df.matter.isin(categories)]
    my_map = df.name.to_dict()
    return {v: k for k, v in my_map.items()}


def process_diagram(params):
    diagram_data_file, label_dict = params
    logger.info(f"Processing: {diagram_data_file}")
    file_hash = diagram_data_file.name.split(".")[0].split("_")[1]
    dss = DiagramSymbolsStorage()
    # Get the pickle information
    metadata = dss.load(file_hash)
    picture_path = os.path.join(
        str(diagram_data_file.parent), diagram_data_file.name.split(".")[0] + ".png"
    )
    # Create the TF Record
    ts = TensorflowStorage()
    tf_record = ts.diagram_to_tf_example(picture_path, metadata, label_dict)
    return tf_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Tensorflow records to train a vision model"
    )
    parser.add_argument(
        "--diagram_matter",
        type=str,
        nargs="*",
        help="""Matters of the diagram, at least two: 'P-Process', 'L-Piping', 'J-Instrument', 'H-HVAC',
            'T-telecom', 'N-Structural', 'R-Mechanical', 'E-Electro', 'S-Safety'""",
        default=None,
    )
    args = parser.parse_args()
    if args.diagram_matter:
        if type(args.diagram_matter) == list:
            diagram_matters = args.diagram_matter
        else:
            diagram_matters = [args.diagram_matter]
    else:
        exit(-1)
    label_map_dict = get_categories_map(diagram_matters)
    logger.info(f"Obtained {len(label_map_dict)} label categories")
    diagram_path = Path(DIAGRAM_PATH)

    params = [(x, label_map_dict) for x in diagram_path.glob("*.pickle")]
    output_path = "/tmp/"
    # Save TF record in chunks
    num_shards = 10
    pool = multiprocessing.Pool(4)
    total_num_annotations_skipped = 0
    writers = [
        tf.python_io.TFRecordWriter(
            output_path + "-%05d-of-%05d.tfrecord" % (i, num_shards)
        )
        for i in range(num_shards)
    ]

    for idx, tf_example in enumerate(pool.imap(process_diagram, params)):
        if idx % 100 == 0:
            logging.info("On image %d of %d", idx, len(params))

        writers[idx % num_shards].write(tf_example.SerializeToString())

    pool.close()
    pool.join()

    for writer in writers:
        writer.close()
