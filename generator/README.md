# Pid Reader Generator

## Objective

This piece of code generates a set of diagrams for ML training purposes.
The diagrams will be a set of machine generated documents with a composition of symbols, text attached to them
and lines connecting those symbols.

With the diagram in PNG version, the system will also output the training data needed to run ML vision models on
top of the generated images.

## Data Pipeline

The data pipeline starts with an ordered collection of symbols placed

### Training for classification

Training dataset generation, in this example, generate 128 diagrams, one symbol per diagram, with the diagram matters
Lpiping and JInstrument and a size of 500x500:

```bash
python generate_set_diagrams.py --number_diagrams 128 --symbols_per_diagram 1 --diagram_matter L-Piping J-Instrument --diagram_size 500 500

python generate_tensorflow.py --diagram_matter L-Piping J-Instrument
```

This will create a folder structure in the data directory /tf/<hexId> with all the files needed for training.

### Training for detection



## Status

Currently

## Dev environment setup

- Install requirements: `pip instal -e .`

- Install extras_require testing packages from setup.cfg:
`pip install pid_reader_generator[testing]`
