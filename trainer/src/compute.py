from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import logging

from azureml.exceptions import ProjectSystemException

logger = logging.getLogger(__name__)


def get_or_create_workspace(
    subscription_id, resource_group, workspace_name, workspace_region
) -> Workspace:
    try:
        ws = Workspace(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
        )
        # write the details of the workspace to a configuration file to the notebook library
        ws.write_config()
        logger.info(
            "Workspace configuration succeeded. Skip the workspace creation steps below"
        )
    except ProjectSystemException as e:
        logger.exception(e)
        # Create the workspace using the specified parameters
        ws = Workspace.create(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            location=workspace_region,
            create_resource_group=True,
            sku="basic",
            exist_ok=True,
        )
        ws.get_details()
        # write the details of the workspace to a configuration file to the notebook library
        ws.write_config()
    return ws


def get_or_create_gpu_cluster(ws, gpu_cluster_name):
    try:
        gpu_cluster = ComputeTarget(workspace=ws, name=gpu_cluster_name)
        logger.info("Found existing gpu cluster")
    except ComputeTargetException:
        logger.info("Creating new gpu-cluster")

        # Specify the configuration for the new cluster
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="STANDARD_NC6", min_nodes=0, max_nodes=4
        )
        # Create the cluster with the specified name and configuration
        gpu_cluster = ComputeTarget.create(ws, gpu_cluster_name, compute_config)

        # Wait for the cluster to complete, show the output log
        gpu_cluster.wait_for_completion(show_output=True)
    return gpu_cluster
