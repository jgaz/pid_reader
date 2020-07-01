from typing import List
from azureml.core import Dataset, Workspace
import logging

from config import LOGGING_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def get_or_create_dataset(
    workspace: Workspace, blob_storage_paths: List[str], dataset_name: str
) -> Dataset:

    try:
        return Dataset.get_by_name(workspace=workspace, name=f"{dataset_name}")
    except Exception:
        logger.info(f"Registering {dataset_name} with {len(blob_storage_paths)} files.")
        dataset = Dataset.File.from_files(path=blob_storage_paths)
        return dataset.register(
            workspace=workspace,
            name=dataset_name,
            description="training and test dataset",
            create_new_version=True,
        )
