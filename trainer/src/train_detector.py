"""
This trains a detector based on the backbone of the given model
The backbone higher layers should be compatible with a B0-Efficientnet

"""
import argparse
import logging
import os
import subprocess
import time
from shutil import copyfile

from typing import Dict

from compute import get_or_create_workspace
from config import (
    LOGGING_LEVEL,
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
    WORKSPACE_REGION,
)
from data import read_training_metadata


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

        command = "protoc --python_out=. ./object_detection/protos/*.proto"
        subprocess.run(command, check=False, cwd="./models/research/", shell=True)
        copyfile(
            "./models/research/object_detection/packages/tf2/setup.py",
            "./models/research/setup.py",
        )
        command = "pip install ."
        subprocess.run(command, check=True, cwd="models/research/", shell=True)


def update_config(config_path: str, variables_to_setup: Dict[str, str]) -> str:
    with open(config_path, "r") as config_file:
        config = config_file.read()

    for key, value in variables_to_setup.items():
        config = config.replace(f"##{key}##", str(value))

    out_file_path = f"{config_path}.custom"
    with open(f"{config_path}.custom", "w") as config_file:
        config_file.write(config)
    return out_file_path


def get_variables(training_path: str, backbone_path: str):
    """
    TODO: cosine_decay_learning_rate figure out how to tweak this
    """
    training_metadata = read_training_metadata(training_path)
    variables = {
        "NUM_CLASSES": int(training_metadata["num_classes"]),
        "DIAGRAM_SIZE": 500,  # Must match backbone training data??
        "BATCH_SIZE": 32,
        "TOTAL_STEPS": int(training_metadata["num_images_training"]) // 32,
        "PATH_LABEL_MAP": os.path.join(training_path, "label_map.pbtxt"),
        "TRAINING_PATH": os.path.join(training_path, "?????-of-000??.tfrecord"),
        "VALIDATION_PATH": os.path.join(
            training_path, "validation", "validation.tfrecord"
        ),
        "BACKBONE_PATH": backbone_path,
    }
    return variables


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training the detector")

    parser.add_argument(
        "--experiment_id", required=True, type=str, help="""The experiment id"""
    )
    parser.add_argument(
        "--training_data_path",
        required=True,
        type=str,
        help="""Where the training data is, in AzureML the mount point""",
    )
    parser.add_argument(
        "--extra_path",
        required=True,
        type=str,
        help="""Where the training data is, rest of the path""",
    )
    parser.add_argument(
        "--object_detection_path",
        required=False,
        type=str,
        default="./models/research/object_detection/",
        help="Where the object detection library is, so it can find the executable",
    )
    parser.add_argument(
        "--backbone_path", type=str, help="Where the backbone model is", default=""
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        help="Where to save the model",
        default="outputs/model",
    )

    args = parser.parse_args()
    experiment_id = args.experiment_id
    training_data_path = args.training_data_path
    extra_path = args.extra_path
    backbone_path = args.backbone_path
    object_detection_path = args.object_detection_path
    output_model_path = args.output_model_path

    # Check for TF2 object detection API or install
    install_tf2_object_detection()

    # Make sure training data is there
    root_folder = os.path.join(training_data_path, extra_path)
    training_folder = os.path.join(root_folder, experiment_id)
    if backbone_path == "":
        backbone_folder = os.path.join(training_folder, "backbone")
    else:
        backbone_folder = os.path.join(training_folder, backbone_path)

    model_dir = os.path.join(output_model_path, f"{experiment_id}")
    os.makedirs(model_dir, exist_ok=True)

    # Update config file information
    path_config = "./deploy/configuration_detector.config"
    config_variables = get_variables(training_folder, backbone_folder)
    config_file_path = update_config(path_config, config_variables)

    object_detection_main_file_path = os.path.join(
        object_detection_path, "model_main_tf2.py"
    )
    # Launch training script
    command = (
        f"python {object_detection_main_file_path}"
        + f" --pipeline_config_path {config_file_path}"
        + f" --model_dir {model_dir}"
        + f" --alsologtostderr"
        + f" --sample_1_of_n_eval_examples 1"
        + f" --checkpoint_every_n 10"
    )

    print(command)
    subprocess.run(command.split(" "), check=True)
