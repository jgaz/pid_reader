"""
Train a model
"""
import argparse
import os
import tensorflow as tf
import tensorflow.keras as tfkeras

from data import read_training_metadata, read_data

from model import ModelFactory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the network")
    parser.add_argument(
        "--experiment_id", required=True, type=str, help="""The experiment id"""
    )
    parser.add_argument(
        "--data_folder", required=True, type=str, help="""Where the training data is""",
    )
    parser.add_argument(
        "--extra_path", required=True, type=str, help="""Where the training data is""",
    )
    parser.add_argument("--epochs", type=str, default=10)

    parser.add_argument(
        "--output_folder",
        type=str,
        help="Where to save the resulting model",
        default="./outputs/",
    )

    args = parser.parse_args()
    experiment_id = args.experiment_id
    data_folder = args.data_folder
    output_folder = args.output_folder
    extra_path = args.extra_path
    epochs = int(args.epochs)

    data_folder = os.path.join(data_folder, extra_path)
    training_metadata = read_training_metadata(data_folder)

    # Strategy to train in multiple GPUs
    strategy = tf.distribute.MirroredStrategy()

    # Open a strategy scope.
    with strategy.scope():
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
    LEARNING_RATE_EXP_DECAY = 0.8  # Set to 0.7 for <500K training set

    # Adjust steps per epoch to the total training samples we have
    steps_per_epoch = training_samples // (BATCH_SIZE * epochs)
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
    model_path = os.path.join(output_folder, experiment_id, "model")
    tensorboard_path = os.path.join(output_folder, experiment_id, "tensorboard")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)

    tensorboard_callback = tfkeras.callbacks.TensorBoard(
        tensorboard_path, update_freq="epoch"
    )
    """
    Check out embeddings metadata :)
    embeddings_freq: frequency (in epochs) at which embedding layers will be visualized. If set to 0,
                     embeddings won't be visualized.
    embeddings_metadata: a dictionary which maps layer name to a file name in which metadata for this
                     embedding layer is saved. See the details about metadata files format. In case if
                     the same metadata file is used for all embedding layers, string can be passed.
    """

    checkpoint_callback = tfkeras.callbacks.ModelCheckpoint(
        os.path.join(model_path, "best_checkpoint"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        verbose=1,
    )

    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[lr_decay, tensorboard_callback, checkpoint_callback],
    )
