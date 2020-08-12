import os
from typing import List

import yaml
from azureml.core import Dataset, Workspace
import tensorflow as tf
import logging

from trainer.config import LOGGING_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


class DataIngestorBackbone:
    def decode_record(self, record):
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

    def normalize_image(self, image):
        """Normalize the image to zero mean and unit variance."""
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        offset = tf.constant([0.485])
        offset = tf.expand_dims(offset, axis=0)
        offset = tf.expand_dims(offset, axis=0)
        image -= offset

        scale = tf.constant([0.229])
        scale = tf.expand_dims(scale, axis=0)
        scale = tf.expand_dims(scale, axis=0)
        image /= scale
        return image

    def decode_image(self, image_bit_string):
        """Decodes the image and set its static shape."""
        image = tf.io.decode_png(image_bit_string, channels=1)
        image = self.normalize_image(image)
        return image

    def select_data_from_record(self, record):
        x = self.decode_image(record["image/encoded"])
        y = record["image/object/class/label"]
        return x, y

    def transform_and_filter(self, ds: tf.data.Dataset):
        """
        Map from example into viable fit input
        """
        ds = ds.map(
            self.decode_record, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        ds = ds.map(
            self.select_data_from_record,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        return ds

    def read_data(
        self, data_folder: str, is_training: bool = True, batch_size: int = 8
    ) -> tf.data.Dataset:
        """

        """
        path = os.path.join(data_folder, "*.tfrecord")
        files_dataset = tf.data.Dataset.list_files(path)
        ds: tf.data.Dataset = tf.data.TFRecordDataset(
            files_dataset,
            compression_type=None,
            buffer_size=None,  # Set it if you need a buffer I/O saving
            num_parallel_reads=None,  # If IO bound, set this > 1
        )

        if is_training:
            ds = ds.shuffle(100)
            ds = ds.repeat()

        self.transform_and_filter(ds)

        if batch_size > 0:
            ds = ds.batch(batch_size, drop_remainder=is_training)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds


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
        training_metadata = yaml.full_load(file)
    return training_metadata


def show_sample(ds: tf.data.Dataset):
    """
    Prints out interesting information about the example used for training
    :param ds: tf.data.Dataset
    """
    inputs = list(ds.take(10).as_numpy_iterator())
    import numpy
    import pandas as pd

    for input in inputs:
        print(
            f"element: image:{input[0].shape} image_min:{numpy.min(input[0])} label:{input[1].shape} label_value:{input[1]}"
        )
        print(pd.value_counts(input[0].flat))
