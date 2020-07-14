"""
Train a model
"""
import argparse
import os
import tensorflow as tf
import tensorflow.keras as tfkeras
from config import MODEL_PATH
from data import read_training_metadata, read_data

from model import ModelFactory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the network")
    parser.add_argument(
        "-e", "--experiment_id", required=True, type=str, help="""The experiment id"""
    )
    parser.add_argument(
        "-d",
        "--data_folder",
        required=True,
        type=str,
        help="""Where the training data is""",
    )
    parser.add_argument(
        "--extra_path", required=True, type=str, help="""Where the training data is""",
    )

    args = parser.parse_args()
    experiment_id = args.experiment_id
    data_folder = args.data_folder
    extra_path = args.extra_path

    data_folder = os.path.join(data_folder, extra_path)
    print(f"Getting data from: {data_folder}")
    training_metadata = read_training_metadata(data_folder)

    model_factory = ModelFactory()
    model = model_factory.get_model(
        training_metadata["width"], training_metadata["num_classes"]
    )
    model.compile(
        optimizer="adam",  # learning rate will be set by LearningRateScheduler
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    # The global batch size will be automatically sharded across all
    # replicas by the tf.data.Dataset API.
    # The best practice is to scale the batch size by the number of
    # replicas (cores). The learning rate should be increased as well.
    training_samples = training_metadata["num_images_training"]
    BATCH_SIZE = 32  # Gobal batch size.
    LEARNING_RATE = 0.01
    LEARNING_RATE_EXP_DECAY = 0.7

    # Adjust 10 epochs to the total training samples we have
    EPOCHS = 10
    steps_per_epoch = training_samples // (BATCH_SIZE * EPOCHS)

    current_model_path = os.path.join(MODEL_PATH, experiment_id)
    training_dataset: tf.data.Dataset = read_data(
        data_folder, is_training=True, batch_size=BATCH_SIZE
    )
    validation_data_folder = os.path.join(data_folder, "validation/")
    validation_dataset: tf.data.Dataset = read_data(
        validation_data_folder, is_training=False, batch_size=BATCH_SIZE
    )

    lr_decay = tfkeras.callbacks.LearningRateScheduler(
        lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY ** epoch, verbose=True
    )
    tensorboard_callback = tfkeras.callbacks.TensorBoard(
        current_model_path, update_freq=100
    )

    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        callbacks=[
            lr_decay,
            tensorboard_callback,
            tfkeras.callbacks.ModelCheckpoint(
                MODEL_PATH,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                mode="min",
                period=1,
            ),
        ],
    )
