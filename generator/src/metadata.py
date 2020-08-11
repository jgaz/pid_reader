import hashlib
import io
import json
import os
import pickle
import shutil

import PIL
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, TypedDict, Any, Dict
from config import METADATA_PATH, DIAGRAM_PATH, DIAGRAM_CLASSES_FILE
import tensorflow.compat.v1 as tf

import logging

from symbol import GenericSymbol

logger = logging.getLogger(__name__)


@dataclass
class SymbolData:
    name: str
    family: str
    description: str
    matter: str


class JsonTrainingObject(TypedDict):
    images: List[str]
    type: str
    annotations: List[Any]
    categories: List[Any]


class SymbolStorage:
    symbols_metadata_file = os.path.join(METADATA_PATH, "symbols.pickle")
    data: pd.DataFrame = None

    def __init__(self):
        try:
            self._read()
        except Exception as e:
            logger.error(
                "Cannot read the source file for symbols, check that you have generated it"
            )
            raise e

    def save(self, symbols: List[Tuple[str, ...]]):
        with open(self.symbols_metadata_file, "wb") as f_out:
            pickle.dump(symbols, f_out)

    def _read(self) -> pd.DataFrame:
        if self.data is None:
            with open(self.symbols_metadata_file, "rb") as f_in:
                symbols = pickle.load(f_in)
            self.data = pd.DataFrame(
                data=symbols, columns=["name", "family", "description", "matter"]
            )
        return self.data

    def _pandas_to_symbol_data(self, df: pd.DataFrame) -> List[SymbolData]:
        symbol_list: List[SymbolData] = []
        for item in list(df.values):
            symbol_list.append(SymbolData(*item))
        return symbol_list

    def get_families(self) -> List[str]:
        return list(self.data.family.unique())

    def get_matters(self) -> List[str]:
        return list(self.data.matter.unique())

    def get_symbols_by_family(self, matter: str, family: str) -> List[SymbolData]:
        df_filtered = self.data.loc[
            (self.data.matter == matter) & (self.data.family == family)
        ]
        return self._pandas_to_symbol_data(df_filtered)

    def get_symbols_by_matter(self, matter: str) -> List[SymbolData]:
        df_filtered = self.data.loc[(self.data.matter == matter)]
        return self._pandas_to_symbol_data(df_filtered)

    def get_dataframe(self):
        return self.data


class BlockedSymbolsStorage:
    blocked_symbols: List[str] = []
    BLOCKED_SYMBOLS_METADATA_FILE = os.path.join(METADATA_PATH, "symbols_blocked.csv")

    def __init__(self):
        self._read()

    def _read(self) -> List[str]:
        if not self.blocked_symbols:
            df = pd.read_csv(self.BLOCKED_SYMBOLS_METADATA_FILE)
            self.blocked_symbols = [x.upper() for x in df.name.values]

        return self.blocked_symbols

    def filter_out_blocked_symbols(
        self, symbols: List[SymbolData], blocked_symbols: List[str]
    ):
        set_blocked_symbols = set([x.upper() for x in blocked_symbols])
        return [s for s in symbols if s.name.upper() not in set_blocked_symbols]


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
    def store_image(self, dss: DiagramSymbolsStorage, image_diagram, diagram_symbols):
        image: PIL.Image = image_diagram.convert("1")
        hash = hashlib.md5(image.tobytes()).hexdigest()
        image.save(os.path.join(DIAGRAM_PATH, f"Diagram_{hash}.png"))
        # Store symbols too
        dss.save(hash, diagram_symbols)

    @staticmethod
    def clear():
        shutil.rmtree(DIAGRAM_PATH)
        os.makedirs(DIAGRAM_PATH)


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

    def get_feature_description(self):
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
            "image/object/class/label": tf.io.VarLenFeature(tf.string),
            "image/object/difficult": tf.io.VarLenFeature(tf.int64),
            "image/object/truncated": tf.io.VarLenFeature(tf.int64),
            "image/object/view": tf.io.VarLenFeature(tf.string),
        }

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
    def reannotate_ids(tf_records):
        idx = 0
        for annotation in tf_records["annotations"]:
            annotation["id"] = idx
            idx += 1


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
