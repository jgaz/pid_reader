import tensorflow.keras as tfkeras
from efficientnet.model import EfficientNet
import collections


class TrainingMetadata:
    pass


class ModelFactory:
    def get_model_detector(self):
        # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
        pass

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
            classes=classes,
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
