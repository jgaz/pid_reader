import argparse

import numpy as np
import os

from generator.metadata import TensorflowStorage
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import preprocess_input

from trainer.config import MODELS_DIRECTORY

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

    x = preprocess_input(image)
    x = np.expand_dims(x, 0)
    y = model.predict(x)

    metadata = TensorflowStorage.load_training_metadata(
        experiment_id=args.experiment_id
    )
    idx = np.argmax(y)
    print(metadata.label_id_mapping[idx])
