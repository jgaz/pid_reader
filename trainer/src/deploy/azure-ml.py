import logging

from compute import (
    get_or_create_workspace,
    get_or_create_gpu_cluster,
)
from __init__ import (
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
    WORKSPACE_REGION,
    GPU_CLUSTER_NAME,
)

logger = logging.getLogger()


if __name__ == "__main__":

    ws = get_or_create_workspace(
        SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, WORKSPACE_REGION
    )
    # Verify that cluster does not exist already
    gpu_cluster = get_or_create_gpu_cluster(ws, GPU_CLUSTER_NAME)
