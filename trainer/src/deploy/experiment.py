import argparse
import logging
from compute import get_or_create_workspace
from config import (
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
    WORKSPACE_REGION,
    LOGGING_LEVEL,
)
from data import get_or_create_dataset
from ml_storage import AzureBlobCloudStorage

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a training experiment in AzureML")
    parser.add_argument("experiment_id", type=str, help="""The experiment id""")
    args = parser.parse_args()
    if args.experiment_id:
        experiment_id = args.experiment_id
    else:
        exit(-1)

    ws = get_or_create_workspace(
        SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, WORKSPACE_REGION
    )

    # Get the list of files from the blob
    ab = AzureBlobCloudStorage()
    files = ab.list_files(experiment_id)
    logger.info(f"Found {len(files)} files")
    # Make sure training dataset exists
    dataset = get_or_create_dataset(ws, files, experiment_id)
