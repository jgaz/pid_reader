# Complete training procedure

This assumes you have set up a symbol library already

## Provision
`python trainer/deploy/azure-ml.py --gpu_machine STANDARD_NC12`

## Training data generation

Classification:
`python generator/generate_set_diagrams.py --number_diagrams 10513
--symbols_per_diagram 1
--diagram_matter L-Piping P-Process J-Instrument
--diagram_size 512 512`

Detector:
`python generator/generate_set_diagrams.py --number_diagrams 10514
--symbols_per_diagram 6
--diagram_matter L-Piping P-Process J-Instrument
--diagram_size 512 512`

## Backbone training
*This takes around 1h when cold starting the environment*


## Detector training
