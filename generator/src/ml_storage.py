import os
import re
from typing import List

from azure.storage.blob import BlobServiceClient


class CloudStorage:
    def store_file(self, file_name, destination_path):
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
            blob_client.upload_blob(data)

    def store_directory(self, path, blob_name):
        for r, d, f in os.walk(path):
            if f:
                for file in f:
                    file_path_on_azure = os.path.join(blob_name, file)
                    file_path_on_local = os.path.join(r, file)
                    self.store_file(file_path_on_local, file_path_on_azure)

    def list_files(self, path: str) -> List[str]:
        container_client = self.blob_service_client.get_container_client(
            container=self.CONTAINER_NAME
        )
        files = container_client.list_blobs(name_starts_with=path)
        blob_path = f"https://{self.storage_account}.core.windows.net/{self.CONTAINER_NAME}/{path}/"
        return [blob_path + x["name"] for x in files]
