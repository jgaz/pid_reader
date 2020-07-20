"""
This analyzes the resulting model passed
"""
import argparse
import os
from typing import Tuple, List, Dict
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from config import GENERATOR_TF_PATH, TRAINED_MODELS_PATH
from data import read_data


def confusion_matrix(experiment_id: str):
    """
    Generate a confusion matrix using the validation dataset
    Outputs a file with the matrix.
    """
    number_of_batches = 300  # This takes a while on a CPU

    batch_size = 32
    test_set, y = load_data(
        experiment_id,
        "validation",
        number_of_batches=number_of_batches,
        batch_size=batch_size,
    )

    model = load_model(experiment_id)
    predictions = model.predict(test_set)

    num_classes = len(predictions[0])
    classes = list(range(num_classes))

    # The confusion matrix is made out of the predictions, so we have to make sure that there are at least one
    # prediction for each class, otherwise the size of the expected matrix will not match and the Pandas dataframe
    # creation will break
    y_pred = np.argmax(predictions, axis=-1)
    con_mat = tf.math.confusion_matrix(labels=y, predictions=y_pred).numpy()
    con_mat_norm = np.around(
        con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis], decimals=2
    )
    return con_mat_norm, classes


def plot_confusion_matrix(con_mat_norm, classes, experiment_id):
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(
        os.path.join(TRAINED_MODELS_PATH, experiment_id, "confusion_matrix.png")
    )


def analyze_confusion_matrix(conf_matrix, label_id_mapping: Dict[int, str]):
    diagonal = conf_matrix.diagonal()
    conf_matrix = np.nan_to_num(conf_matrix)
    non_zero_per_row = [
        np.nonzero(conf_matrix[x, :]) for x in range(conf_matrix.shape[0])
    ]

    report = []
    for idx, element in enumerate(diagonal):
        report.append(
            {
                "class": label_id_mapping[idx],
                "accuracy": float(element),
                "confused with:": [
                    label_id_mapping[x] for x in non_zero_per_row[idx][0]
                ],
            }
        )
    return report


def load_model(experiment_id: str) -> tf.keras.models.Model:
    model = tf.keras.models.load_model(os.path.join(TRAINED_MODELS_PATH, experiment_id))
    return model


def load_data(
    experiment_id: str, folder: str, number_of_batches: int = 10, batch_size: int = 32
) -> Tuple[tf.data.Dataset, List[int]]:
    validation_data_folder = os.path.join(GENERATOR_TF_PATH, experiment_id, folder)
    validation_dataset: tf.data.Dataset = read_data(
        validation_data_folder, is_training=False, batch_size=batch_size
    )

    validation_dataset = validation_dataset.take(number_of_batches)

    unbatched = validation_dataset.unbatch()
    y = [int(x[1][0]) for x in unbatched]

    return validation_dataset, y


def load_training_metadata(experiment_id):
    training_metadata_file = os.path.join(
        GENERATOR_TF_PATH, experiment_id, "training_metadata.yaml"
    )
    return yaml.full_load(open(training_metadata_file, "r"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the network")
    parser.add_argument("experiment_id", type=str, help="""The experiment id""")
    args = parser.parse_args()
    training_metadata = load_training_metadata(experiment_id=args.experiment_id)
    label_id_mapping = training_metadata["label_id_mapping"]
    normal_confusion_matrix, classes = confusion_matrix(args.experiment_id)
    plot_confusion_matrix(normal_confusion_matrix, classes, args.experiment_id)
    report = analyze_confusion_matrix(normal_confusion_matrix, label_id_mapping)
    report_file = os.path.join(
        TRAINED_MODELS_PATH, args.experiment_id, "confusion.yaml"
    )
    with open(report_file, "w") as file:
        yaml.dump(report, file)
