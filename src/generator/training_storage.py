import os
from typing import List, Tuple, Dict
import tensorflow.compat.v1 as tf
import yaml
import io
import pickle
import shutil
import hashlib
import json
import PIL
import logging

from generator.config import GENERATOR_METADATA_FILE, DIAGRAM_PATH, DIAGRAM_CLASSES_FILE
from generator.metadata import JsonTrainingObject, SymbolData
from generator.symbol import GenericSymbol
from trainer.config import GENERATOR_TF_PATH

logger = logging.getLogger(__name__)


class TensorflowStorage:
    """
    {
        "images": [
            {"file_name": "2008_000008.jpg", "height": 442, "width": 500, "id": 1},
        ],
        "type": "instances",
        "annotations": [
            {
                "area": 139194,                "iscrowd": 0,                "image_id": 1,
                "bbox": [53, 87, 418, 333],                "category_id": 13,                "id": 1,
                "ignore": 0,                "segmentation": [],
            },
        ],
        "categories": [{"supercategory": "none", "id": 1, "name": "aeroplane"},],
    }
    """

    def diagram_to_tf_example(
        self,
        full_path: str,
        metadata: List[GenericSymbol],
        label_map_dict,
        file_idx: int,
    ) -> Tuple[tf.train.Example, JsonTrainingObject]:
        """Convert pickle file into tfrecord

        Notice that this function normalizes the bounding box coordinates provided
        by the raw data.

        Args:
            full_path: name of the generated diagram
            metadata: metadata of the diagram
            label_map_dict: A map from symbol names to integers ids.
            file_idx: annotation json dictionary.

        Returns:
          example: The converted tf.Example.

        Raises:
          ValueError: if the image pointed to by data['filename'] is not a valid JPEG
        """
        ann_json_dict: JsonTrainingObject = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": [],
        }

        with tf.gfile.GFile(full_path, "rb") as fid:
            original_encoded_img = fid.read()
        encoded_img_io = io.BytesIO(original_encoded_img)
        image = PIL.Image.open(encoded_img_io)
        key = hashlib.sha256(original_encoded_img).hexdigest()
        width, height = image.size
        file_name = full_path.split("/")[-1]
        image_id = file_idx

        if ann_json_dict:
            image = {
                "file_name": file_name,
                "height": height,
                "width": width,
                "id": image_id,
            }
            ann_json_dict["images"].append(image)

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        poses: List[str] = []
        difficult_obj = []

        for symbol in metadata:
            difficult_obj.append(int(False))  # Not needed
            truncated.append(int(False))  # Not truncated either

            xmin.append(float(symbol.x) / width)
            ymin.append(float(symbol.y) / height)
            xmax.append(float(symbol.x + symbol.size_w) / width)
            ymax.append(float(symbol.y + symbol.size_h) / height)
            classes_text.append(symbol.name.encode("utf8"))
            classes.append(label_map_dict[symbol.name])
            if ann_json_dict:
                abs_xmin = int(symbol.x)
                abs_ymin = int(symbol.y)
                abs_xmax = int(symbol.x + symbol.size_w)
                abs_ymax = int(symbol.y + symbol.size_h)
                abs_width = abs_xmax - abs_xmin
                abs_height = abs_ymax - abs_ymin
                ann = {
                    "area": abs_width * abs_height,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [abs_xmin, abs_ymin, abs_width, abs_height],
                    "category_id": label_map_dict[symbol.name],
                    "id": 0,
                    "ignore": 0,
                    "segmentation": [],
                }
                ann_json_dict["annotations"].append(ann)

        example = tf.train.Example(
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
        return example, ann_json_dict

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _bytes_list_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def get_feature_description():
        return {
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/width": tf.io.FixedLenFeature([], tf.int64),
            "image/filename": tf.io.VarLenFeature(tf.string),
            "image/source_id": tf.io.VarLenFeature(tf.string),
            "image/key/sha256": tf.io.VarLenFeature(tf.string),
            "image/encoded": tf.io.VarLenFeature(tf.string),
            "image/format": tf.io.VarLenFeature(tf.string),
            "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
            "image/object/class/text": tf.io.VarLenFeature(tf.string),
            "image/object/class/label": tf.io.VarLenFeature(tf.int64),
            "image/object/difficult": tf.io.VarLenFeature(tf.int64),
            "image/object/truncated": tf.io.VarLenFeature(tf.int64),
            "image/object/view": tf.io.VarLenFeature(tf.string),
        }

    @staticmethod
    def get_sample_from_dataset(filenames: List[str]):
        raw_dataset = tf.data.TFRecordDataset(filenames)
        raw_record = raw_dataset.take(1)
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

    @staticmethod
    def parse_dataset(filenames: List[str]):
        raw_dataset = tf.data.TFRecordDataset(filenames)

        # Create a dictionary describing the features.
        feature_description = TensorflowStorage.get_feature_description()

        def _parse_image_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, feature_description)

        return raw_dataset.map(_parse_image_function)

    @staticmethod
    def reannotate_ids(tf_records):
        idx = 0
        for annotation in tf_records["annotations"]:
            annotation["id"] = idx
            idx += 1

    @staticmethod
    def load_training_metadata(experiment_id):
        training_metadata_file = os.path.join(
            GENERATOR_TF_PATH, experiment_id, GENERATOR_METADATA_FILE
        )
        return yaml.full_load(open(training_metadata_file, "r"))


class TrainingDatasetLabelDictionaryStorage:
    @staticmethod
    def save(valid_symbols: List[SymbolData]):
        logger.info("Saving symbols dictionary")
        valid_symbols_dict = {}
        for i, symbol in enumerate(valid_symbols):
            valid_symbols_dict[symbol.name] = i + 1
        file_path = os.path.join(DIAGRAM_PATH, DIAGRAM_CLASSES_FILE)
        json.dump(valid_symbols_dict, open(file_path, "w"))

    @staticmethod
    def get(data_path: str) -> Dict[str, int]:
        """
        Get the object names and Ids
        :param data_path: path of the class file
        :return:
        """
        classes_filename = os.path.join(data_path, DIAGRAM_CLASSES_FILE)
        dictionary = json.load(open(classes_filename, "r"))
        return dictionary


class DiagramSymbolsStorage:
    """
    Storage of the symbol metadata: position, type, etc...
    """

    PATH = DIAGRAM_PATH

    def _get_path(self, hash: str):
        return os.path.join(DiagramSymbolsStorage.PATH, f"Diagram_{hash}.pickle")

    def save(self, hash: str, symbols: List[GenericSymbol]):
        pickle.dump(symbols, open(self._get_path(hash), "wb"))

    def load(self, hash: str = None, filename: str = None):
        if filename:
            return pickle.load(open(filename, "rb"))
        elif hash:
            return pickle.load(open(self._get_path(hash), "rb"))


class DiagramStorage:
    """
    Store the diagram created
    """

    def store_image(self, dss: DiagramSymbolsStorage, image_diagram, diagram_symbols):
        image: PIL.Image = image_diagram.convert("1")
        hash = hashlib.md5(image.tobytes()).hexdigest()
        image.save(os.path.join(DIAGRAM_PATH, f"Diagram_{hash}.png"))
        # Store symbols too
        dss.save(hash, diagram_symbols)

    @staticmethod
    def clear():
        """
        Clear the directory where diagrams have been generated
        """
        shutil.rmtree(DIAGRAM_PATH)
        os.makedirs(DIAGRAM_PATH)
