import os
import logging

# ML Settings
SUBSCRIPTION_ID = os.getenv(
    "SUBSCRIPTION_ID", default="9bc2f845-5f0d-450d-bf32-82d81d9e8445"
)
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP", default="jgazStudentRG")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME", default="mlvision")
WORKSPACE_REGION = os.getenv("WORKSPACE_REGION", default="westeurope")
GPU_CLUSTER_NAME = "gpu-cluster"
ENVIRONMENT_NAME_DETECTOR = "detector"

SRC_PATH = os.path.dirname(os.path.realpath(__file__))
MODELS_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(SRC_PATH)), "data_trainer"
)
TRAINED_MODELS_PATH = os.path.join(SRC_PATH, MODELS_DIRECTORY)
GENERATOR_PATH = os.path.join(SRC_PATH, "../../generator/")
GENERATOR_TF_PATH = os.path.join(GENERATOR_PATH, "../data_generator/tf/")
GENERATOR_METADATA_FILE = "training_metadata.yaml"

LOGGING_LEVEL = logging.INFO
