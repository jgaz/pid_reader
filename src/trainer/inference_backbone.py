import argparse

import numpy as np
import os

from generator.training_storage import TensorflowStorage
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from trainer.config import MODELS_DIRECTORY
from trainer.data import DataIngestorBackbone

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference in EfficientNet")
    parser.add_argument(
        "--experiment_id", type=str, help="""The model file""", required=True
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
    model_path = os.path.join(
        MODELS_DIRECTORY, experiment_id, experiment_id, "model/best_checkpoint"
    )
    model = load_model(model_path)

    image = imread(image_file)
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.show()

    di = DataIngestorBackbone()
    with tf.io.gfile.GFile(image_file, "rb") as fid:
        original_encoded_img = fid.read()
        x = di.decode_image(original_encoded_img)

    y = model.predict(np.array([x.numpy(),]))

    metadata = TensorflowStorage.load_training_metadata(
        experiment_id=args.experiment_id
    )
    idx = np.argmax(y)
    print(f"Index: {idx} symbol:{metadata['label_id_mapping'][idx]}")
