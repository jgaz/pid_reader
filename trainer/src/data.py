import os
from typing import List

import yaml
from azureml.core import Dataset, Workspace
import tensorflow as tf
import logging

from config import LOGGING_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def get_or_create_dataset(
    workspace: Workspace, blob_storage_paths: List[str], dataset_name: str
) -> Dataset:

    try:
        return Dataset.get_by_name(workspace=workspace, name=f"{dataset_name}")
    except Exception:
        logger.info(f"Registering {dataset_name} with {len(blob_storage_paths)} files.")
        dataset = Dataset.File.from_files(path=blob_storage_paths)
        return dataset.register(
            workspace=workspace,
            name=dataset_name,
            description="training and test dataset",
            create_new_version=True,
        )


def read_training_metadata(training_path: str):
    yaml_file_path = os.path.join(training_path, "training_metadata.yaml")
    with open(yaml_file_path, "r") as file:
        training_metadata = yaml.load(file)
    return training_metadata


def read_data(
    data_folder: str, is_training: bool = True, batch_size: int = 8
) -> tf.data.Dataset:
    files_dataset = tf.data.Dataset.list_files(f"{data_folder}*.tfrecord")
    ds: tf.data.Dataset = tf.data.TFRecordDataset(
        files_dataset,
        compression_type=None,
        buffer_size=None,  # Set it if you need a buffer I/O saving
        num_parallel_reads=None,  # If IO bound, set this > 1
    )

    if is_training:
        ds = ds.shuffle(100)
        ds = ds.repeat()

    # Map from example into viable fit input
    def _decode_record(record):
        feature = {
            "image/filename": tf.io.FixedLenFeature((), tf.string),
            "image/format": tf.io.FixedLenFeature((), tf.string),
            "image/key/sha256": tf.io.FixedLenFeature((), tf.string),
            "image/encoded": tf.io.FixedLenFeature((), tf.string),
            "image/source_id": tf.io.FixedLenFeature((), tf.string, ""),
            "image/height": tf.io.FixedLenFeature((), tf.int64, -1),
            "image/width": tf.io.FixedLenFeature((), tf.int64, -1),
            "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
            "image/object/class/label": tf.io.VarLenFeature(tf.int64),
            "image/object/is_crowd": tf.io.VarLenFeature(tf.int64),
            "image/object/class/text": tf.io.VarLenFeature(tf.string),
            "image/object/difficult": tf.io.VarLenFeature(tf.int64),
            "image/object/truncated": tf.io.VarLenFeature(tf.int64),
            "image/object/view": tf.io.VarLenFeature(tf.string),
        }

        parsed_tensors = tf.io.parse_single_example(record, feature)

        # Turn tensors in to dense so it can be fed into the model
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=""
                    )
                else:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=0
                    )
        return parsed_tensors

    def _decode_image(record):
        """Decodes the image and set its static shape."""
        image = tf.io.decode_image(record["image/encoded"], channels=1)
        # image.set_shape([record['image/width'], record['image/height'], 1])
        return image

    def _select_data_from_record(record):
        x = _decode_image(record=record)
        y = record["image/object/class/label"]
        return (x, y)

    ds = ds.map(_decode_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(
        _select_data_from_record, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=is_training)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
