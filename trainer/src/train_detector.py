"""
This trains a detector based on the backbone of the given model
"""
import argparse
import os
import subprocess
from shutil import copyfile


def install_tf2_object_detection():
    try:
        import pexpect
    except ImportError:
        # Clone the tensorflow models repository if it doesn't already exist
        command = "git  clone --depth 1 https://github.com/tensorflow/models".split(" ")
        subprocess.run(command, check=True)

        command = "protoc object_detection/protos/*.proto --python_out=.".split(" ")
        subprocess.run(command, check=True, cwd="./models/research/")

        copyfile(
            "./models/research/object_detection/packages/tf2/setup.py", "./setup.py"
        )
        command = "python -m pip install .".split(" ")
        subprocess.run(command, check=True)

        import pexpect


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

    data_folder = os.path.join(data_folder, extra_path)

    # Check for TF2 object detection API or install
    install_tf2_object_detection()

    # Download backbone model

    # Make sure training data is there
    # Launch training script
