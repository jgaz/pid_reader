# Pid Reader Generator

## Objective

This piece of code generates a synthetic dataset of diagrams for ML training purposes. The main reason to this
synthetic approach is the limitation on labeled data, the process is tedious and requires skilled people
so the aim is to create good enough "fake" diagrams to train a model.

The diagrams will be a set of machine generated documents with a composition of symbols, text attached to them
and lines connecting those symbols, with some overlap and 90 degrees rotations.

With the diagram in PNG version, the system will also output the training data needed to run ML vision models on
top of the generated images.


## Data Pipeline

The data pipeline starts with a collection of symbols used to create a synthetic data set for the training
of the network.

In order to train a EfficientNet + EfficientDet networks (check out the trainer Readme for more details on the
machine learning side) we need to create two sets of data:
- EfficientNet: This is a image classification network, thus the input is an image and a class for the object.
- EfficientDet: This is an object detection that could be built on top of an already trained EfficientNet backbone. The inputs
for this are:
  - Image, the same size as the classification algorithm.
  - Boxes where objects have been detected with the coordinates and the class.


### Dataset for classification

The classification network needs one symbol per diagram so it can learn to identify the class. In the real world the symbol
does not come just by itself, it is surrounded by text, may be other symbols and lines. In addition it can be presented
rotated 90 degrees and its size does vary.

All this behaviour has been implemented in the generator with the exception of the lines, we need more experimentation around
it, a priori it seems that the lines shouldn't confuse the classification task.

The resulting dataset will be uploaded to an AzureBlob configured in the environment variable:
`AZURE_STORAGE_CONNECTION_STRING`

In this example, we generate 128 diagrams, one symbol per diagram, with the diagram matters
Lpiping and JInstrument and a size of 500x500:

```bash
python generate_set_diagrams.py --number_diagrams 128 --symbols_per_diagram 1 --diagram_matter L-Piping J-Instrument --diagram_size 500 500
```

This will create a folder structure in the data directory /tf/<hexId> with all the files needed for training, rougly:
- Tensorflow files for the data in binary format, for both training and validation.
- Yaml file containing metadata about the training dataset.
- Json file containing metadata about the training examples generated.

For production purposes, using 500K examples yields 95% accuracy with around 400 classes, we will need around 6 hours of 2xK80 GPUs.


### Dataset for detection

The dataset for detection is currently under development, more details yet to come.


## Dev environment setup

- Install requirements: `pip instal -e .`

- Install extras_require testing packages from setup.cfg:
`pip install pid_reader_generator[testing]`
