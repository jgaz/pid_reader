"""
Adapted from: https://colab.research.google.com/drive/1sLqFKVV94wm-lglFq_0kGo2ciM0kecWD#scrollTo=-Ycfl7rnDT1D&uniqifier=1
"""
import argparse
from io import BytesIO

import numpy as np
import os

from PIL import Image
from generator.metadata import TensorflowStorage
from object_detection.builders import model_builder

import matplotlib.pyplot as plt
import tensorflow as tf
from trainer.config import MODELS_DIRECTORY, GENERATOR_TF_PATH
from trainer.model import ObjectDetectionConfigurator
from object_detection.utils import config_util, label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection import config_checkpoint


def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, "rb").read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 1)).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference in EfficientNet")
    parser.add_argument(
        "--experiment_id", type=str, help="""The model file""", required=True
    )
    parser.add_argument(
        "--backbone_id", type=str, help="""Model backbone file""", required=True
    )
    parser.add_argument(
        "--checkpoint", type=str, help="""Model checkpoint file""", required=True
    )
    parser.add_argument(
        "--image_file",
        type=str,
        help="The image to test against the model",
        required=True,
    )

    args = parser.parse_args()
    image_file = args.image_file
    experiment_id = args.experiment_id
    backbone_id = args.backbone_id
    checkpoint_name = args.checkpoint

    # Set up the labels to use API
    metadata = TensorflowStorage.load_training_metadata(
        experiment_id=args.experiment_id
    )
    label_mapping = metadata["label_id_mapping"]
    category_index = {}
    for key, name in label_mapping.items():
        category_index[key] = {"id": key, "name": name}

    # recover our saved model
    odc = ObjectDetectionConfigurator()
    path_config = "./deploy/configuration_detector.config"
    training_path = os.path.join(GENERATOR_TF_PATH, experiment_id)
    backbone_path = os.path.join(
        MODELS_DIRECTORY, backbone_id, backbone_id, "model/best_checkpoint"
    )

    config_file_path = odc.update_config(
        path_config, training_path=training_path, backbone_path=backbone_path
    )

    pipeline_config = config_file_path
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs["model"]
    # Update
    config_checkpoint.FINE_TUNE_CHECKPOINT = configs[
        "train_config"
    ].fine_tune_checkpoint

    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    checkpoint_path = os.path.join(
        MODELS_DIRECTORY,
        f"detector_{experiment_id}/model/{experiment_id}/{checkpoint_name}",
    )
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(checkpoint_path)

    def get_model_detection_function(model):
        """Get a tf.function for detection."""

        @tf.function
        def detect_fn(image):
            """Detect objects in image."""
            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            detections = model.postprocess(prediction_dict, shapes)
            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn

    detect_fn = get_model_detection_function(detection_model)

    # Do inference
    image_np = load_image_into_numpy_array(image_file)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    # Present results
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections["detection_boxes"][0].numpy(),
        (detections["detection_classes"][0].numpy() + label_id_offset).astype(int),
        detections["detection_scores"][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.5,
        agnostic_mode=False,
    )
    plt.figure(figsize=(8, 8))
    plt.imshow(image_np_with_detections)
    plt.show()
