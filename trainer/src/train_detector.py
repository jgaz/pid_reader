"""
This trains a detector based on the backbone of the given model
"""
import argparse
import logging
import os
import subprocess
from shutil import copyfile

from typing import Dict

from config import LOGGING_LEVEL
from data import read_training_metadata

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def install_tf2_object_detection():
    try:
        import object_detection
    except ImportError:
        # Clone the tensorflow models repository if it doesn't already exist
        command = "git clone --depth 1 https://github.com/tensorflow/models".split(" ")
        subprocess.run(command, check=False)

        command = "protoc object_detection/protos/*.proto --python_out=.".split(" ")
        subprocess.run(command, check=False, cwd="models/research/")

        copyfile(
            "./models/research/object_detection/packages/tf2/setup.py",
            "./models/research/setup.py",
        )
        command = "python -m pip install .".split(" ")
        subprocess.run(command, check=True, cwd="models/research/")


def update_config(config_path: str, variables_to_setup: Dict[str, str]):
    with open(config_path, "r") as config_file:
        config = config_file.read()

    for key, value in variables_to_setup.items():
        config.replace(f"##{key}##", value)

    with open(config_path, "w") as config_file:
        config_file.write(config)


def get_variables(training_path: str):
    training_metadata = read_training_metadata(training_path)
    variables = {
        "DIAGRAM_SIZE": int(training_metadata["width"]),
        "BATCH_SIZE": 32,
        "TOTAL_STEPS": int(training_metadata["num_images_training"]) // 32,
        "PATH_BACKBONE": os.path.join(training_path, "backbone/saved_model.pb"),
        "PATH_LABEL_MAP": os.path.join(training_path, "label_map.pbtxt"),
        "TRAINING_PATH": os.path.join(training_path, "?????-of-000??.tfrecord"),
        "VALIDATION_PATH": os.path.join(
            training_path, "validation", "validation.tfrecord"
        ),
    }
    return variables


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

    args = parser.parse_args()
    experiment_id = args.experiment_id
    data_folder = args.data_folder
    extra_path = args.extra_path

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
    path_config = "./deploy/configuration_detector.config"
    model_dir = f"./outputs/model/{experiment_id}"
    config_variables = get_variables(training_folder)
    print(config_variables)
    # update_config(path_config)

    # Launch training script
    command = f"""python ./content/models/research/object_detection/model_main_tf2.py
    --pipeline_config_path={path_config} --model_dir={model_dir}
    --alsologtostderr --sample_1_of_n_eval_examples=1 """
    print(command)
"""
python train_detector.py --experiment_id 21dc09821e6e4b722b93878a078977483ba798dd \
 --data_folder ./ \
 --extra_path https/storageaccountdatav9498.blob.core.windows.net/pub/

"""
