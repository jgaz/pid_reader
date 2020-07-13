import os
import logging

# ML Settings
SUBSCRIPTION_ID = os.getenv(
    "SUBSCRIPTION_ID", default="9bc2f845-5f0d-450d-bf32-82d81d9e8445"
)
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP", default="jgazStudentRG")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME", default="mlvision")
WORKSPACE_REGION = os.getenv("WORKSPACE_REGION", default="northeurope")
GPU_CLUSTER_NAME = "gpu-cluster"


SRC_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(SRC_PATH), "data", "models")

LOGGING_LEVEL = logging.INFO
