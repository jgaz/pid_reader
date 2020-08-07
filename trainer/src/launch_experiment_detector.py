import argparse
import logging
import os

from azureml.core import Run, Environment, Workspace, Experiment
from azureml.train.dnn import TensorFlow

from compute import get_or_create_workspace, get_or_create_detector_environment
from config import (
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
    WORKSPACE_REGION,
    LOGGING_LEVEL,
    MODELS_DIRECTORY,
    GPU_CLUSTER_NAME,
)
from data import get_or_create_dataset
from ml_storage import AzureBlobCloudStorage, ExperimentStorage

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def copy_backbone_model(experiment_id: str, backbone_experiment_id: str):
    """
    Copies the model stored locally into a backbone folder
    in the blob storage for the training data
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
    parser.add_argument(
        "--run_local",
        type=bool,
        help="Whether to run the computation in local computer",
        default=False,
    )

    args = parser.parse_args()
    experiment_id = args.experiment_id
    run_local = args.run_local
    backbone_experiment_id = args.backbone_experiment_id

    compute_target = GPU_CLUSTER_NAME
    if run_local:
        compute_target = "local"

    ws: Workspace = get_or_create_workspace(
        SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, WORKSPACE_REGION
    )

    env: Environment = get_or_create_detector_environment(ws, force_creation=True)

    # Get the list of files from the blob
    ab = AzureBlobCloudStorage()
    files = ab.list_files(experiment_id)

    # We need to add the backbone model to the list of files
    files += copy_backbone_model(experiment_id, backbone_experiment_id)

    # Make sure training dataset exists
    dataset_name = f"detector_{experiment_id}"
    dataset = get_or_create_dataset(ws, files, dataset_name)

    # Create the experiment
    experiment_name = f"detector_{experiment_id}"
    experiment = Experiment(ws, name=experiment_name)

    blob_storage_path = f"https/{ab.storage_account}.blob.core.windows.net/pub"

    script_params = {
        "--training_data_path": dataset.as_named_input(dataset_name).as_download(),
        "--extra_path": blob_storage_path,
        "--experiment_id": experiment_id,
    }
    script_folder = "./"

    estimator = TensorFlow(
        source_directory=script_folder,
        compute_target=compute_target,
        script_params=script_params,
        entry_script="train_detector.py",
        framework_version="2.1",
        environment_definition=env,
    )
    logger.info("Starting experiment")
    run: Run = experiment.submit(estimator)

    run.wait_for_completion(show_output=True)

    logger.info("Experiment ended, downloading model")
    es = ExperimentStorage(ws, experiment_id)
    es.download_output(run)
    logger.info("All done!")
