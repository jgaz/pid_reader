import argparse
import logging

from compute import (
    get_or_create_workspace,
    get_or_create_gpu_cluster,
)
from config import (
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
    WORKSPACE_REGION,
    GPU_CLUSTER_NAME,
)


logger = logging.getLogger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a training experiment in AzureML")
    parser.add_argument(
        "--gpu_machine",
        type=str,
        help="""The GPU machine, by default is: STANDARD_NC6""",
        default="STANDARD_NC6",
    )
    args = parser.parse_args()
    ws = get_or_create_workspace(
        SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, WORKSPACE_REGION
    )

    # Verify that cluster does not exist already
    gpu_cluster = get_or_create_gpu_cluster(ws, GPU_CLUSTER_NAME, args.gpu_machine)
