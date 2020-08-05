import argparse

from azureml.core import Workspace

from compute import get_or_create_workspace
from config import SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, WORKSPACE_REGION
from ml_storage import ExperimentStorage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the latest outputs of the given experiment"
    )
    parser.add_argument("experiment_id", type=str, help="""The experiment id""")
    args = parser.parse_args()
    experiment_id = args.experiment_id
    ws: Workspace = get_or_create_workspace(
        SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, WORKSPACE_REGION
    )
    experiment_storage = ExperimentStorage(ws, experiment_id)
    experiment_storage.download_output()
