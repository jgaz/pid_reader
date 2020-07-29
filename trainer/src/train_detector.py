"""
This trains a detector based on the backbone of the given model

Example:
    python train_detector.py --experiment_id 21dc09821e6e4b722b93878a078977483ba798dd \
        --data_folder ./ \
        --extra_path https/storageaccountdatav9498.blob.core.windows.net/pub/

"""
import argparse
import logging
import os
import subprocess
from shutil import copyfile

from typing import Dict

from config import LOGGING_LEVEL
from data import read_training_metadata
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def install_tf2_object_detection():
    """
    This will install object detection API into the production cluster
    if you are doing development, you will prefer to install it using
    pip install -e . so it is in development mode
    (you might need to change parts of the API)
    """
    try:
        import object_detection
    except ImportError:
        # Clone the modified tensorflow models repository if it doesn't already exist
        command = "git clone --depth 1 https://github.com/jgaz/models".split(" ")
        subprocess.run(command, check=False)

        command = "protoc object_detection/protos/*.proto --python_out=.".split(" ")
        subprocess.run(command, check=False, cwd="models/research/")

        copyfile(
            "./models/research/object_detection/packages/tf2/setup.py",
            "./models/research/setup.py",
        )
        command = "python -m pip install .".split(" ")
        subprocess.run(command, check=True, cwd="models/research/")


def update_config(config_path: str, variables_to_setup: Dict[str, str]) -> str:
    with open(config_path, "r") as config_file:
        config = config_file.read()

    for key, value in variables_to_setup.items():
        config = config.replace(f"##{key}##", str(value))

    out_file_path = f"{config_path}.custom"
    with open(f"{config_path}.custom", "w") as config_file:
        config_file.write(config)
    return out_file_path


def get_variables(training_path: str):
    """
    TODO: cosine_decay_learning_rate figure out how to tweak this
    """
    training_metadata = read_training_metadata(training_path)
    variables = {
        "NUM_CLASSES": int(training_metadata["num_classes"]),
        "DIAGRAM_SIZE": 500,  # Must match backbone training data??
        "BATCH_SIZE": 32,
        "TOTAL_STEPS": int(training_metadata["num_images_training"]) // 32,
        "PATH_BACKBONE": os.path.join(training_path, "backbone/checkpoint/"),
        "PATH_LABEL_MAP": os.path.join(training_path, "label_map.pbtxt"),
        "TRAINING_PATH": os.path.join(training_path, "?????-of-000??.tfrecord"),
        "VALIDATION_PATH": os.path.join(
            training_path, "validation", "validation.tfrecord"
        ),
    }
    return variables


def generate_backbone_checkpoint(backbone_path: str):
    target_dir = os.path.join(backbone_path, "checkpoint/")
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        model = tf.keras.models.load_model(backbone_path)
        # Save an object based checkpoint
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.save(file_prefix=target_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training the detector")

    parser.add_argument(
        "--experiment_id", required=True, type=str, help="""The experiment id"""
    )
    parser.add_argument(
        "--data_folder", required=True, type=str, help="""Where the training data is""",
    )
    parser.add_argument(
        "--extra_path", required=True, type=str, help="""Where the training data is""",
    )
    parser.add_argument(
        "--object_detection_path",
        required=False,
        type=str,
        default="./models/research/object_detection/",
    )

    args = parser.parse_args()
    experiment_id = args.experiment_id
    data_folder = args.data_folder
    extra_path = args.extra_path
    object_detection_path = args.object_detection_path

    # Check for TF2 object detection API or install
    install_tf2_object_detection()

    # Make sure training data is there
    root_folder = os.path.join(data_folder, extra_path)
    training_folder = os.path.join(root_folder, experiment_id)
    backbone_folder = os.path.join(training_folder, "backbone")

    logger.info(
        f"root {root_folder} training_folder {training_folder} backbone_folder {backbone_folder}"
    )

    # Update config file information
    path_config = os.path.join(data_folder, "deploy/configuration_detector.config")
    model_dir = os.path.join(data_folder, f"outputs/model/{experiment_id}")
    config_variables = get_variables(training_folder)

    config_file_path = update_config(path_config, config_variables)

    generate_backbone_checkpoint(backbone_folder)

    object_detection_main_file_path = os.path.join(
        object_detection_path, "model_main_tf2.py"
    )
    # Launch training script
    command = f"""python {object_detection_main_file_path} --pipeline_config_path={config_file_path} --model_dir={model_dir} --alsologtostderr --sample_1_of_n_eval_examples=1"""
    print(command)
    subprocess.run(command.split(" "), check=True)
