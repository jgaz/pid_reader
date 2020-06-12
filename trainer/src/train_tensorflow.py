from compute_resources import (
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
from azureml.core.dataset import Dataset

# https://docs.microsoft.com/en-gb/azure/machine-learning/how-to-train-tensorflow

if __name__ == "__main__":
    ws = get_or_create_workspace(
        SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, WORKSPACE_REGION
    )
    # Verify that cluster does not exist already
    gpu_cluster = get_or_create_gpu_cluster(ws, GPU_CLUSTER_NAME)

    # Get or create the dataset
    web_paths = [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    ]

    dataset = Dataset.File.from_files(path=web_paths)
    dataset = dataset.register(
        workspace=ws,
        name="pascal-dataset",
        description="training and test pascal dataset",
        create_new_version=True,
    )

    # Prepare the directories of the training data
    # http://localhost:8888/notebooks/how-to-use-azureml/ml-frameworks/tensorflow/training/train-tensorflow-resume-training/train-tensorflow-resume-training.ipynb

    # Create a tensorflow estimator

    from azureml.train.dnn import TensorFlow
    from azureml.core import Experiment

    experiment_name = "tf-resume-training"
    experiment = Experiment(ws, name=experiment_name)

    script_params = {"--data-folder": dataset.as_named_input("mnist").as_mount()}

    # Estimator help: https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/training-with-deep-learning
    script_folder = ""
    compute_target = ""
    estimator = TensorFlow(
        source_directory=script_folder,
        compute_target=compute_target,
        script_params=script_params,
        entry_script="tf_mnist_with_checkpoint.py",
        use_gpu=True,
        pip_packages=["azureml-dataprep[pandas,fuse]"],
    )

    run = experiment.submit(estimator)
    print(run)

    # Tensorboard for the training
    # https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/training-with-deep-learning/export-run-history-to-tensorboard/export-run-history-to-tensorboard.ipynb

    # Configure native distributed training
    # https://docs.microsoft.com/en-gb/azure/machine-learning/how-to-train-tensorflow#distributed-training
