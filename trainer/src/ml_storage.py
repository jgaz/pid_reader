import os
import re
from typing import List

from azure.storage.blob import BlobServiceClient
from azureml.core import Experiment, Workspace, Run

from config import TRAINED_MODELS_PATH
import logging

logger = logging.getLogger(__name__)


class CloudStorage:
    def store_file(self, file_name, destination_path):
        pass

    def store_directory(self, path, blob_name):
        pass

    def list_files(self, path: str) -> List[str]:
        pass


class AzureBlobCloudStorage(CloudStorage):
    CONTAINER_NAME = "pub"
    # Storage account for training data
    TRAINING_STORAGE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING") or exit(
        "AZURE_STORAGE_CONNECTION_STRING needed"
    )

    def __init__(self):
        self.connect_str = self.TRAINING_STORAGE_CONN_STR
        self.storage_account = re.search(r"AccountName=(.*?);", self.connect_str).group(
            1
        )
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connect_str
        )

    def store_file(self, file_path: str, destination_path: str):
        # Create a blob client using the local file name as the name for the blob
        blob_client = self.blob_service_client.get_blob_client(
            container=self.CONTAINER_NAME, blob=destination_path
        )

        # Upload the created file
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    def store_directory(self, path: str, blob_path: str):
        """
        Stores a directory
        """
        for r, d, f in os.walk(path):
            if f:
                for file in f:
                    local_relative_path = r.replace(path, "")
                    local_relative_path = (
                        local_relative_path[1:]
                        if local_relative_path.startswith("/")
                        else local_relative_path
                    )
                    # Terrible hack for validation folder
                    if re.findall(
                        "validation", r
                    ):  # Move into folder if named validation
                        file_path_on_azure = os.path.join(blob_path, "validation", file)
                    else:
                        file_path_on_azure = os.path.join(
                            blob_path, local_relative_path, file
                        )

                    file_path_on_local = os.path.join(r, file)
                    self.store_file(file_path_on_local, file_path_on_azure)

    def list_files(self, path: str) -> List[str]:
        container_client = self.blob_service_client.get_container_client(
            container=self.CONTAINER_NAME
        )
        files = container_client.list_blobs(name_starts_with=path)
        blob_path = f"https://{self.storage_account}.blob.core.windows.net/{self.CONTAINER_NAME}/"
        return [blob_path + x["name"] for x in files]


class ExperimentStorage:
    def __init__(self, workspace: Workspace, experiment_id: str):
        self.experiment_id = experiment_id
        self.experiment = Experiment(workspace, experiment_id)

    def download_output(self, experiment_run=None):
        if experiment_run is None:
            experiment_run: Run = next(self.experiment.get_runs())
        model_path = os.path.join(TRAINED_MODELS_PATH, self.experiment_id)
        logger.info(f"Downloading results in {model_path}")
        os.makedirs(model_path, exist_ok=True)
        experiment_run.download_files("outputs/", model_path, append_prefix=False)
