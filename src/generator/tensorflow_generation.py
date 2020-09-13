"""

"""
import hashlib
import os
from pathlib import Path
import json
import yaml
from typing import Dict, Tuple, List

from generator.config import (
    DIAGRAM_PATH,
    LOGGING_LEVEL,
    TENSORFLOW_PATH,
    CPU_COUNT,
    GENERATOR_METADATA_FILE,
    GENERATOR_LABEL_FILE,
)
import logging
from generator.metadata import JsonTrainingObject
import multiprocessing
import tensorflow.compat.v1 as tf
from generator.training_storage import (
    DiagramSymbolsStorage,
    TensorflowStorage,
    TrainingDatasetLabelDictionaryStorage,
)

from trainer.ml_storage import AzureBlobCloudStorage

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


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


def save_metadata_yaml(
    json_annotation, label_map_dict, output_path, num_shards, diagram_matters, model_id
):
    """
    Saves a yaml file detailing metadata for the training dataset used.
    :param json_annotation:
    :param label_map_dict:
    :param output_path:
    :param num_shards:
    :param diagram_matters:
    :param model_id:
    :return:
    """
    logger.info(f"Obtained {len(json_annotation['images'])} images in json")
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
    yaml_file_path = os.path.join(output_path, GENERATOR_METADATA_FILE)
    with open(yaml_file_path, "w") as file:
        yaml.dump(yaml_additional_data_contents, file)


def save_metadata_label_map(output_path: str, label_map_dict: Dict[str, int]):
    """
    Generate ProtocolBuffer file with the classes map for the detector
    :param output_path:
    :param label_map_dict:
    :return:
    """
    from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap

    label_map = StringIntLabelMap()
    with open(os.path.join(output_path, GENERATOR_LABEL_FILE), "wb") as f:
        for name, id in label_map_dict.items():
            label_item = label_map.item.add()
            label_item.name = name
            label_item.id = id
        f.write(str(label_map).encode())


def generate_train_dataset(diagram_matters: List[str]) -> str:
    """
    Obtains all the diagrams from the data directory and stores them in appropriate
    tensorflow format to train models. It saves training metadata file too.

    :param diagram_matters:
    :return:
    """

    label_map_dict = TrainingDatasetLabelDictionaryStorage.get(DIAGRAM_PATH)
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

    model_id = hashlib.sha1(str.encode(f"{diagram_matters}{len(params)}")).hexdigest()
    output_path = os.path.join(TENSORFLOW_PATH, model_id)
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    # Save TF record in chunks
    num_shards = 20
    pool = multiprocessing.Pool(CPU_COUNT)
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

    save_metadata_yaml(
        json_annotation=json_annotation,
        label_map_dict=label_map_dict,
        output_path=output_path,
        num_shards=num_shards,
        diagram_matters=diagram_matters,
        model_id=model_id,
    )
    save_metadata_label_map(output_path, label_map_dict)

    cs = AzureBlobCloudStorage()
    cs.store_directory(output_path, model_id)

    return model_id
