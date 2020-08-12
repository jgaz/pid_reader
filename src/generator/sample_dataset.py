import argparse

from generator.metadata import TensorflowStorage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obtain a sample of the tensorflow dataset"
    )
    parser.add_argument(
        "--dataset_file", type=str, required=True, help="File holding the tfRecord",
    )
    args = parser.parse_args()
    dataset_file = args.dataset_file
    parsed_dataset = TensorflowStorage.parse_dataset(dataset_file)
    for training_example in parsed_dataset:
        print(training_example)
        exit(0)
