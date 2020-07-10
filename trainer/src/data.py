import os
from typing import List

import yaml
from azureml.core import Dataset, Workspace
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


def read_data():
    """
    tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": self._int64_feature(height),
                    "image/width": self._int64_feature(width),
                    "image/filename": self._bytes_feature(file_name.encode("utf8")),
                    "image/source_id": self._bytes_feature(
                        str(image_id).encode("utf8")
                    ),
                    "image/key/sha256": self._bytes_feature(key.encode("utf8")),
                    "image/encoded": self._bytes_feature(original_encoded_img),
                    "image/format": self._bytes_feature("png".encode("utf8")),
                    "image/object/bbox/xmin": self._float_list_feature(xmin),
                    "image/object/bbox/xmax": self._float_list_feature(xmax),
                    "image/object/bbox/ymin": self._float_list_feature(ymin),
                    "image/object/bbox/ymax": self._float_list_feature(ymax),
                    "image/object/class/text": self._bytes_list_feature(classes_text),
                    "image/object/class/label": self._int64_list_feature(classes),
                    "image/object/difficult": self._int64_list_feature(difficult_obj),
                    "image/object/truncated": self._int64_list_feature(truncated),
                    "image/object/view": self._bytes_list_feature(poses),
                }
            )
        )
    :return:
    """
    pass
