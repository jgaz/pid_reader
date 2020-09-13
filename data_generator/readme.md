# Training generation

## Symbol database

The symbol database is stored in `symbols.csv` file, it is organized
hierarchically in the following way:

1. matter: this is the main grouping field, it can be used to generate
training examples only with symbols of the wanted matters.
2. family: subgroup of matter, right now it is not in use.
3. name: this should be unique as the examples are labeled using the
name. The symbols PNG files should be named after the value of this
field: "name.png", the name is case sensitive.

Finally there is another field called description that is currently
of no use.

### Blocking symbols

If your database has symbols that you don't want to use for training,
just put the names in the csv and they will be ignored.

### Text configuration

Symbols are usually surrounded by text, some times the text is placed
within the symbol, this file configures where should be put the text
if the symbol allows it.

The fields are:
    - name: should match with name in the symbol database
    - max_lines: maximum number of lines
    - x: x coord of top left corner
    - y: y coord of top left corner
    - resol: resolution to use

## Symbol files

Symbols are stored in the folder `/symbol_libraries/png`.
In this library you can find three folders containing the symbols in the
different resolutions: 100, 225 and 600.

The format is PNG stored in Grayscale or black and white.

## Training examples generation

The training examples are generated and stored in `data_generator/tf/<datasetID>`
The dataset id is just an MD5 of a string with the number of examples and the matters selected
to create the training set.

The generation is a two step process:
1. `launch_generate_set_diagrams.py` this will create all the diagrams and their metadata
in `data_generator\diagrams` this folder will be cleared with the script start and then
it will hold the number of examples required (one png file and one pickle with the metadata).
2.  Finally this script will pack all the images and metadata into a suitable
tensorflow record, the format is compatible with the training material for EfficientDet. The last step of
this script is to push the content into a Azure Blob Storage.

Example:
```bash
launch_generate_set_diagrams.py  --number_diagrams 100000 \
    --symbols_per_diagram 10 \
    --diagram_matter P-Process,L-Piping \
    --diagram_size 512 512
```
