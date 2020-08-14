import argparse
import logging
import os

from azureml.train.dnn import TensorFlow
from azureml.core import Experiment, Run, Environment

from trainer.compute import get_or_create_workspace, get_or_create_detector_environment
from trainer.config import (
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
    WORKSPACE_REGION,
    LOGGING_LEVEL,
    GPU_CLUSTER_NAME,
)
from trainer.data import get_or_create_dataset
from trainer.ml_storage import AzureBlobCloudStorage, ExperimentStorage

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the backbone training experiment in AzureML"
    )
    parser.add_argument("experiment_id", type=str, help="""The experiment id""")
    parser.add_argument("--epochs", type=str, default=10)
    parser.add_argument(
        "--run_local",
        type=bool,
        help="Whether to run the computation in local computer",
        default=False,
    )

    args = parser.parse_args()
    experiment_id = args.experiment_id
    epochs = args.epochs
    run_local = args.run_local

    compute_target = GPU_CLUSTER_NAME
    if run_local:
        compute_target = "local"

    ws = get_or_create_workspace(
        SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, WORKSPACE_REGION
    )

    env: Environment = get_or_create_detector_environment(ws, force_creation=True)

    # Get the list of files from the blob
    ab = AzureBlobCloudStorage()
    files = ab.list_files(experiment_id)

    # Make sure training dataset exists
    dataset_name = f"a_{experiment_id}"
    dataset = get_or_create_dataset(ws, files, dataset_name)

    # Create the experiment
    experiment_name = experiment_id
    experiment = Experiment(ws, name=experiment_name)

    script_params = {
        "--data_folder": dataset.as_named_input(dataset_name).as_download(),
        "--extra_path": os.path.join(
            f"https%3A/%2F{ab.storage_account}.blob.core.windows.net/pub",
            f"{experiment_id}/",  # Seems a bug in Azure SDK
        ),
        "--experiment_id": f"{experiment_id}",
        "--epochs": f"{epochs}",
    }

    script_folder = "./"
    estimator = TensorFlow(
        source_directory=script_folder,
        compute_target=compute_target,
        script_params=script_params,
        entry_script="trainer/train_backbone.py",
        framework_version="2.1",
        environment_definition=env,
    )

    run: Run = experiment.submit(estimator)

    run.wait_for_completion(show_output=True)

    es = ExperimentStorage(ws, experiment_id)
    es.download_output(run)

    # Monitoring experiments
    # https://docs.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments
