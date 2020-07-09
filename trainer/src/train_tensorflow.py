import argparse
from typing import List

import tensorflow.keras as tfkeras

from model import ModelFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the network")
    parser.add_argument("experiment_id", type=str, help="""The experiment id""")
    parser.add_argument("data_folder", type=str, help="""Where the training data is""")

    args = parser.parse_args()
    experiment_id = args.experiment_id
    data_folder = args.data_folder
    print(f"Arguments: {experiment_id} {data_folder}")

    model_factory = ModelFactory()
    model = model_factory.get_model(500, 100)
    print("Model retrieved")
    model.compile(
        optimizer="adam",  # learning rate will be set by LearningRateScheduler
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # The global batch size will be automatically sharded across all
    # replicas by the tf.data.Dataset API.
    # The best practice is to scale the batch size by the number of
    # replicas (cores). The learning rate should be increased as well.
    TRAINING_SAMPLES = 1000
    BATCH_SIZE = 64  # Gobal batch size.
    LEARNING_RATE = 0.01
    LEARNING_RATE_EXP_DECAY = 0.7
    EPOCHS = 10
    steps_per_epoch = TRAINING_SAMPLES // BATCH_SIZE

    MODEL_PATH = ""
    training_dataset: List = []

    lr_decay = tfkeras.callbacks.LearningRateScheduler(
        lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY ** epoch, verbose=True
    )

    tensorboard_callback = tfkeras.callbacks.TensorBoard(MODEL_PATH, update_freq=100)

    history = model.fit(
        training_dataset,
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
