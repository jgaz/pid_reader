"""

"""
import argparse
import hashlib
import os
from pathlib import Path
import json
import yaml
from typing import Dict, Tuple
from config import DIAGRAM_PATH, LOGGING_LEVEL, TENSORFLOW_PATH, CPU_COUNT
import logging
from metadata import (
    DiagramSymbolsStorage,
    TensorflowStorage,
    JsonTrainingObject,
)
import multiprocessing
import tensorflow.compat.v1 as tf

from ml_storage import AzureBlobCloudStorage

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


def get_categories_map(data_path: str) -> Dict[str, int]:
    # Get the object names and Ids
    classes_filename = os.path.join(data_path, "classes.json")
    dictionary = json.load(open(classes_filename, "r"))
    return dictionary


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


def save_metadata_yaml(json_annotation, label_map_dict, output_path, num_shards):
    logger.info(
        f"Processed {idx + 1} files, obtained {len(json_annotation['images'])} images in json"
    )
    logger.info(f"Obtained {len(json_annotation['annotations'])} annotations in json")
    size = (
        json_annotation["images"][0]["width"],
        json_annotation["images"][0]["height"],
    )
    # Save Json file
    TensorflowStorage.reannotate_ids(json_annotation)
    json_file_path = output_path + "/json_pascal.json"
    with tf.io.gfile.GFile(json_file_path, "w") as f:
        json.dump(json_annotation, f)
    total_images = len(json_annotation["images"])
    images_per_shard = total_images // num_shards
    # Create YAML file with training metadata
    yaml_additional_data_contents = {
        "num_images_validation": images_per_shard,
        "num_images_training": total_images - images_per_shard,
        "matters": diagram_matters,
        "model_id": model_id,
        "num_classes": len(label_map_dict),
        "label_id_mapping": {v: k for k, v in label_map_dict.items()},
        "height": size[0],
        "width": size[1],
    }
    yaml_file_path = output_path + "/training_metadata.yaml"
    with open(yaml_file_path, "w") as file:
        yaml.dump(yaml_additional_data_contents, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Tensorflow records to train a vision model"
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
    args = parser.parse_args()
    if args.diagram_matter:
        if type(args.diagram_matter) == list:
            diagram_matters = args.diagram_matter
        else:
            diagram_matters = [args.diagram_matter]
    else:
        exit(-1)
    diagram_path = Path(DIAGRAM_PATH)
    label_map_dict = get_categories_map(str(diagram_path))
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

    params = [
        (file_idx, file_name, label_map_dict)
        for file_idx, file_name in enumerate(diagram_path.glob("*.pickle"))
    ]

    model_id = hashlib.sha1(str.encode(f"{diagram_matters}{len(params)}")).hexdigest()
    output_path = os.path.join(TENSORFLOW_PATH, model_id)
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    # Save TF record in chunks
    num_shards = 20
    pool = multiprocessing.Pool(CPU_COUNT)
    total_num_annotations_skipped = 0
    files_out = [
        output_path + "/%05d-of-%05d.tfrecord" % (i, num_shards)
        for i in range(num_shards)
    ]
    writers = [tf.python_io.TFRecordWriter(f) for f in files_out]
    for idx, tf_process in enumerate(pool.imap(process_diagram, params)):
        tf_example, pool_json_annotation = tf_process
        if idx % 1000 == 0:
            logging.info("On image %d of %d", idx, len(params))
        merge_json_annotations(json_annotation, pool_json_annotation)
        writers[idx % num_shards].write(tf_example.SerializeToString())
    pool.close()
    pool.join()
    for writer in writers:
        writer.close()

    # Spare one chunk for validation
    validation_path = os.path.join(output_path, "validation")
    try:
        os.mkdir(validation_path)
    except FileExistsError:
        pass
    validation_filename = os.path.join(validation_path, "validation.tfrecord")
    os.replace(
        output_path + "/%05d-of-%05d.tfrecord" % (num_shards - 1, num_shards),
        validation_filename,
    )

    save_metadata_yaml(json_annotation, label_map_dict, output_path, num_shards)

    cs = AzureBlobCloudStorage()
    cs.store_directory(output_path, model_id)
