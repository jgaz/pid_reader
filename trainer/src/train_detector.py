"""
This trains a detector based on the backbone of the given model
"""
import argparse
import os
import subprocess
from shutil import copyfile


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


def update_config(config_path: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the detector")
    parser.add_argument(
        "--backbone_experiment_id",
        required=True,
        type=str,
        help="""The backbone experiment id""",
    )
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
    backbone_experiment_id = args.backbone_experiment_id
    experiment_id = args.experiment_id
    data_folder = args.data_folder
    extra_path = args.extra_path

    # Check for TF2 object detection API or install
    install_tf2_object_detection()

    # Make sure training data is there
    root_folder = os.path.join(data_folder, extra_path)
    training_folder = os.path.join(root_folder, experiment_id)
    backbone_folder = os.path.join(root_folder, backbone_experiment_id)

    # Update config file information
    path_config = "../deploy/configuration_detector.config"
    update_config(path_config)

    # Launch training script
    command = """python /content/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_file} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps={num_eval_steps}"""
