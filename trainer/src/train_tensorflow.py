"""
Train a model
"""
import argparse
import os
import tensorflow as tf
import tensorflow.keras as tfkeras
from azureml.core import Run

from config import MODEL_PATH, TENSORBOARD_PATH
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
    BATCH_SIZE = 64  # Gobal batch size.
    LEARNING_RATE = 0.01
    LEARNING_RATE_EXP_DECAY = 0.7

    # Adjust 10 epochs to the total training samples we have
    EPOCHS = 10
    steps_per_epoch = training_samples // (BATCH_SIZE * EPOCHS)
    training_dataset: tf.data.Dataset = read_data(
        data_folder, is_training=True, batch_size=BATCH_SIZE
    )
    validation_data_folder = os.path.join(data_folder, "validation/")
    validation_dataset: tf.data.Dataset = read_data(
        validation_data_folder, is_training=False, batch_size=BATCH_SIZE
    )

    # Logger
    run = Run.get_context()

    lr_decay = tfkeras.callbacks.LearningRateScheduler(
        lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY ** epoch, verbose=True
    )

    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(TENSORBOARD_PATH, exist_ok=True)

    tensorboard_callback = tfkeras.callbacks.TensorBoard(
        TENSORBOARD_PATH, update_freq="epoch"
    )
    """
    Check out embeddings metadata :)
    embeddings_freq: frequency (in epochs) at which embedding layers will be visualized. If set to 0, embeddings won't be visualized.
    embeddings_metadata: a dictionary which maps layer name to a file name in which metadata for this embedding layer is saved. See the details about metadata files format. In case if the same metadata file is used for all embedding layers, string can be passed.
    """

    checkpoint_callback = tfkeras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        verbose=1,
    )

    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        callbacks=[lr_decay, tensorboard_callback, checkpoint_callback],
    )

    # We may not need the history since we are using the tensorboard callback
