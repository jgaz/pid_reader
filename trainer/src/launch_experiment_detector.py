import argparse
import logging
import os

from azureml.train.dnn import TensorFlow
from azureml.core import Experiment, Run, Environment, Workspace

from compute import get_or_create_workspace, get_or_create_detector_environment
from config import (
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
    WORKSPACE_REGION,
    LOGGING_LEVEL,
    GPU_CLUSTER_NAME,
    MODELS_DIRECTORY,
)
from data import get_or_create_dataset
from ml_storage import AzureBlobCloudStorage

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def run_details(run: Run):
    print(run.get_details())


def get_model(run: Run, experiment_id: int):
    model_path = f"./{MODELS_DIRECTORY}/{experiment_id}"
    os.makedirs(model_path, exist_ok=True)

    for f in run.get_file_names():
        if f.startswith("outputs/model"):
            output_file_path = os.path.join(model_path, f.split("/")[-1])
            logger.info("Downloading from {} to {} ...".format(f, output_file_path))
            run.download_file(name=f, output_file_path=output_file_path)


def copy_backbone_model(experiment_id: str, backbone_experiment_id: str):
    """
    Copies the model into a backbone folder in the blob storage for the training data
    """
    cs = AzureBlobCloudStorage()
    local_path = os.path.join(MODELS_DIRECTORY, backbone_experiment_id)
    blob_path = os.path.join(experiment_id, "backbone")
    cs.store_directory(local_path, blob_path=blob_path)
    return cs.list_files(blob_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a training experiment in AzureML")
    parser.add_argument("experiment_id", type=str, help="""The experiment id""")
    parser.add_argument(
        "--backbone_experiment_id",
        required=True,
        type=str,
        help="""The backbone model experiment id""",
    )

    args = parser.parse_args()
    experiment_id = args.experiment_id
    backbone_experiment_id = args.backbone_experiment_id

    ws: Workspace = get_or_create_workspace(
        SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, WORKSPACE_REGION
    )

    env: Environment = get_or_create_detector_environment(ws)

    # Get the list of files from the blob
    ab = AzureBlobCloudStorage()
    files = ab.list_files(experiment_id)

    # We need to add the backbone model to the list of files
    files += copy_backbone_model(experiment_id, backbone_experiment_id)

    # Make sure training dataset exists
    dataset_name = f"detector_{experiment_id}"
    dataset = get_or_create_dataset(ws, files, dataset_name)

    """
    # Create the experiment
    experiment_name = f"detector_{experiment_id}"
    experiment = Experiment(ws, name=experiment_name)

    script_params = {
        "--data_folder": "../data",
        "--extra_path": f"https/{ab.storage_account}.blob.core.windows.net/pub",
        "--experiment_id": experiment_id,
        "--backbone_experiment_id": backbone_experiment_id,
    }
    script_folder = "./"

    estimator = TensorFlow(
        source_directory=script_folder,
        compute_target=GPU_CLUSTER_NAME,
        script_params=script_params,
        entry_script="train_detector.py",
        framework_version="2.1",
        environment_definition=env,
    )

    run: Run = experiment.submit(estimator)

    run.wait_for_completion(show_output=True)

    # run_details(run)

    # get_model(run, experiment_id)

    # Monitoring experiments
    # https://docs.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments
    """
