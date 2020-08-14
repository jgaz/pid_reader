import os

import tensorflow.keras as tfkeras
from efficientnet.model import EfficientNet
from trainer.data import read_training_metadata
import collections


class ModelFactory:

    """
    EXPERIMENT ADDING PADDING LAYERS
    does not work as EfficientDet is not a Sequential model

    def find_layer_index_by_name(self, name: str, model: tfkeras.Model):
        for idx, layer in enumerate(model.layers):
            if layer.name == name:
                return idx
        raise Exception(f"Layer {name} not found")

    def insert_padding_layer(self, model: tfkeras.Model, layer_name: str, padding: Tuple):
        #  Inserts a padding/shrinking layer in a given model for compatibility in downstream models
        layer_idx = self.find_layer_index_by_name(layer_name, model) + 1
        layers = model.layers[:layer_idx] + \
                 [tfkeras.layers.ZeroPadding2D(padding=padding, name=f"{layer_name}_padded")] + \
                 [tfkeras.layers.Cropping2D(cropping=padding, name=f"{layer_name}_cropped")] + \
                 model.layers[layer_idx:]
        new_model = tfkeras.Sequential(layers)

        return new_model

    def insert_compatibility_layers(self, model: tfkeras.Model):
        padding = ((0, 1), (0, 1))
        new_model = self.insert_padding_layer(model, "block3b_add", padding)
        return new_model

    def get_model_detector(self):
        # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
        pass
    """

    def get_model(self, image_size: int, classes: int) -> EfficientNet:
        shape_one_channel = (image_size, image_size, 1)

        model = EfficientNet(
            width_coefficient=0.7,
            depth_coefficient=0.7,
            default_resolution=image_size,
            dropout_rate=0.15,
            drop_connect_rate=0.15,
            depth_divisor=8,
            blocks_args=self._get_block_args(),
            model_name="efficientnet-t01",
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=shape_one_channel,
            pooling=None,
            classes=classes
            + 1,  # Model assumes class 0, no match exists, add it as it is not in the labels dictionary
            backend=tfkeras.backend,
            layers=tfkeras.layers,
            models=tfkeras.models,
            utils=tfkeras.utils,
        )
        return model

    def _get_block_args(self):

        block_args_type = collections.namedtuple(
            "BlockArgs",
            [
                "kernel_size",
                "num_repeat",
                "input_filters",
                "output_filters",
                "expand_ratio",
                "id_skip",
                "strides",
                "se_ratio",
            ],
        )
        # defaults will be a public argument for namedtuple in Python 3.7
        # https://docs.python.org/3/library/collections.html#collections.namedtuple
        block_args_type.__new__.__defaults__ = (None,) * len(block_args_type._fields)

        """
        # B0 base
        # expand_ratio: multiplier of the number of input_filters
        DEFAULT_BLOCKS_ARGS = [
                BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
                          expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
                BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
                          expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
                BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
                          expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
                BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
                          expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
                BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
                          expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
                BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
                          expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
                BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
                          expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
        ]
        """
        default_blocks_args = [
            block_args_type(
                kernel_size=3,
                num_repeat=1,
                input_filters=32,
                output_filters=16,
                expand_ratio=1,
                id_skip=True,
                strides=[1, 1],
                se_ratio=0.25,
            ),
            block_args_type(
                kernel_size=3,
                num_repeat=2,
                input_filters=16,
                output_filters=24,
                expand_ratio=4,
                id_skip=True,
                strides=[2, 2],
                se_ratio=0.25,
            ),
            block_args_type(
                kernel_size=5,
                num_repeat=2,
                input_filters=24,
                output_filters=40,
                expand_ratio=4,
                id_skip=True,
                strides=[2, 2],
                se_ratio=0.25,
            ),
            block_args_type(
                kernel_size=3,
                num_repeat=3,
                input_filters=40,
                output_filters=80,
                expand_ratio=4,
                id_skip=True,
                strides=[2, 2],
                se_ratio=0.25,
            ),
            block_args_type(
                kernel_size=5,
                num_repeat=3,
                input_filters=80,
                output_filters=112,
                expand_ratio=4,
                id_skip=True,
                strides=[1, 1],
                se_ratio=0.25,
            ),
            block_args_type(
                kernel_size=5,
                num_repeat=4,
                input_filters=112,
                output_filters=192,
                expand_ratio=4,
                id_skip=True,
                strides=[2, 2],
                se_ratio=0.25,
            ),
            block_args_type(
                kernel_size=3,
                num_repeat=1,
                input_filters=192,
                output_filters=320,
                expand_ratio=4,
                id_skip=True,
                strides=[1, 1],
                se_ratio=0.25,
            ),
        ]
        return default_blocks_args


class ObjectDetectionConfigurator:
    def update_config(
        self, config_path: str, training_path: str, backbone_path: str
    ) -> str:
        with open(config_path, "r") as config_file:
            config = config_file.read()

        variables_to_setup = self.get_variables(training_path, backbone_path)
        for key, value in variables_to_setup.items():
            config = config.replace(f"##{key}##", str(value))

        out_file_path = f"{config_path}.custom"
        with open(f"{config_path}.custom", "w") as config_file:
            config_file.write(config)
        return out_file_path

    def get_variables(self, training_path: str, backbone_path: str):
        training_metadata = read_training_metadata(training_path)
        variables = {
            "NUM_CLASSES": int(training_metadata["num_classes"]),
            "DIAGRAM_SIZE": int(
                training_metadata["height"]
            ),  # Must match backbone training data
            "BATCH_SIZE": 16,
            "TOTAL_STEPS": int(training_metadata["num_images_training"]) // 16,
            "PATH_LABEL_MAP": os.path.join(training_path, "label_map.pbtxt"),
            "TRAINING_PATH": os.path.join(training_path, "?????-of-000??.tfrecord"),
            "VALIDATION_PATH": os.path.join(
                training_path, "validation", "validation.tfrecord"
            ),
            "BACKBONE_PATH": backbone_path,
        }
        return variables
