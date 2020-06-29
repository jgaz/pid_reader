import os

from azure.storage.blob import BlobServiceClient


class CloudStorage:
    def store_file(self, file_name, destination_path):
        pass


class AzureBlobCloudStorage(CloudStorage):
    CONTAINER_NAME = "pub"

    def __init__(self):
        self.connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING") or exit(
            "AZURE_STORAGE_CONNECTION_STRING needed"
        )

    def store_file(self, file_path, destination_path):
        # Create a blob client using the local file name as the name for the blob
        blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
        blob_client = blob_service_client.get_blob_client(
            container=self.CONTAINER_NAME, blob=file_path
        )

        # Upload the created file
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data)
