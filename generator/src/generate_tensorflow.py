"""

"""
import argparse
import hashlib
import os
from pathlib import Path
import json
import yaml
from typing import List, Dict, Tuple
from config import DIAGRAM_PATH, LOGGING_LEVEL, TENSORFLOW_PATH
import logging
from metadata import (
    DiagramSymbolsStorage,
    TensorflowStorage,
    SymbolStorage,
    JsonTrainingObject,
)
import multiprocessing
import tensorflow.compat.v1 as tf

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


def get_categories_map(categories: List[str]) -> Dict[str, int]:
    # Get the object names and Ids
    ss = SymbolStorage()
    df = ss.get_dataframe()
    df = df[df.matter.isin(categories)]
    my_map = df.name.to_dict()
    # Reverse the mapping, resulting in name->Id
    return {v: k for k, v in my_map.items()}


def process_diagram(params) -> Tuple[tf.train.Example, JsonTrainingObject]:
    diagram_data_file_idx, diagram_data_file, label_dict = params
    file_hash = diagram_data_file.name.split(".")[0].split("_")[1]
    dss = DiagramSymbolsStorage()
    # Get the pickle information
    metadata = dss.load(file_hash)
    picture_path = os.path.join(
        str(diagram_data_file.parent), diagram_data_file.name.split(".")[0] + ".png"
    )
    # Create the TF Record
    ts = TensorflowStorage()
    tf_record, json_annotations = ts.diagram_to_tf_example(
        picture_path, metadata, label_dict, diagram_data_file_idx
    )
    return tf_record, json_annotations


def merge_json_annotations(full, partial):
    full["images"] += partial["images"]
    full["annotations"] += partial["annotations"]


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

    json_annotation: JsonTrainingObject = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [
            {"supercategory": "none", "id": k, "name": v}
            for k, v in label_map_dict.items()
        ],
    }

    diagram_path = Path(DIAGRAM_PATH)
    params = [
        (file_idx, file_name, label_map_dict)
        for file_idx, file_name in enumerate(diagram_path.glob("*.pickle"))
    ]

    model_id = hashlib.sha256(str.encode(f"{diagram_matters}{len(params)}")).hexdigest()
    output_path = os.path.join(TENSORFLOW_PATH, model_id)
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    # Save TF record in chunks
    num_shards = 10
    pool = multiprocessing.Pool(4)
    total_num_annotations_skipped = 0
    writers = [
        tf.python_io.TFRecordWriter(
            output_path + "/%05d-of-%05d.tfrecord" % (i, num_shards)
        )
        for i in range(num_shards)
    ]

    for idx, tf_process in enumerate(pool.imap(process_diagram, params)):
        tf_example, pool_json_annotation = tf_process
        if idx % 100 == 0:
            logging.info("On image %d of %d", idx, len(params))
        merge_json_annotations(json_annotation, pool_json_annotation)
        writers[idx % num_shards].write(tf_example.SerializeToString())

    pool.close()
    pool.join()
    for writer in writers:
        writer.close()

    logger.info(
        f"Processed {idx + 1} files, obtained {len(json_annotation['images'])} images in json"
    )
    logger.info(f"Obtained {len(json_annotation['annotations'])} annotations in json")

    # Save Json file
    TensorflowStorage.reannotate_ids(json_annotation)
    json_file_path = output_path + "/json_pascal.json"
    with tf.io.gfile.GFile(json_file_path, "w") as f:
        json.dump(json_annotation, f)

    # Create the YAML model finetune config file
    yaml_config_contents = {
        "num_classes": len(label_map_dict),
        "var_freeze_expr": "(efficientnet|fpn_cells|resample_p6)",
        "label_id_mapping": {v: k for k, v in label_map_dict.items()},
    }
    yaml_file_path = output_path + "/config_finetune.yaml"
    with open(yaml_file_path, "w") as file:
        documents = yaml.dump(yaml_config_contents, file)
    # Create YAML file with training metadata
    yaml_additional_data_contents = {
        "num_images": len(json_annotation["images"]),
        "matters": diagram_matters,
        "model_id": model_id,
    }
    yaml_file_path = output_path + "/training_metadata.yaml"
    with open(yaml_file_path, "w") as file:
        documents = yaml.dump(yaml_additional_data_contents, file)
