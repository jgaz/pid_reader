from azureml.core import Workspace, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import logging

from azureml.exceptions import ProjectSystemException

from trainer.config import ENVIRONMENT_NAME_DETECTOR

logger = logging.getLogger(__name__)


def get_or_create_workspace(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    workspace_region: str,
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


def get_or_create_gpu_cluster(
    ws: Workspace, gpu_cluster_name: str, gpu_machine: str
) -> ComputeTarget:
    try:
        gpu_cluster = ComputeTarget(workspace=ws, name=gpu_cluster_name)
        logger.info("Found existing gpu cluster")
    except ComputeTargetException:
        logger.info("Creating new gpu-cluster")

        # Specify the configuration for the new cluster
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=gpu_machine,
            min_nodes=0,
            max_nodes=1,
            idle_seconds_before_scaledown=300,
        )
        # Create the cluster with the specified name and configuration
        gpu_cluster = ComputeTarget.create(ws, gpu_cluster_name, compute_config)

        # Wait for the cluster to complete, show the output log
        gpu_cluster.wait_for_completion(show_output=True)
    return gpu_cluster


def get_or_create_detector_environment(
    ws: Workspace, force_creation=False
) -> Environment:
    def _create_environment(ws, environment_name):
        env = Environment(workspace=ws, name=environment_name)
        env.docker.enabled = True
        env.docker.base_image = None
        env.docker.base_dockerfile = open("./Dockerfile.detector", "r").read()
        env.python.user_managed_dependencies = True
        env.register(workspace=ws)
        return env

    if not force_creation:
        try:
            return Environment.get(ENVIRONMENT_NAME_DETECTOR)
        except Exception:
            pass

    return _create_environment(ws, ENVIRONMENT_NAME_DETECTOR)
