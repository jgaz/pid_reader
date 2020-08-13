# Complete training procedure

This assumes you have set up a symbol library already

## Provision
`python trainer/deploy/azure-ml.py --gpu_machine STANDARD_NC12`

## Training data generation

Classification
```bash
python generator/generate_set_diagrams.py \
  --number_diagrams 10513 \
  --symbols_per_diagram 1 \
  --diagram_matter L-Piping P-Process J-Instrument \
  --diagram_size 512 512
```

Detector
We want 512x512 images, with 6 symbols per image and the symbol groups
L-Piping P-Process J-Instrument (defined in the symbol library).

```bash
python generator/generate_set_diagrams.py \
  --number_diagrams 10514 \
  --symbols_per_diagram 6 \
  --diagram_matter L-Piping P-Process J-Instrument \
  --diagram_size 512 512
```

## Backbone training
This takes around 1h when cold starting the environment, mainly
due to creating the custom docker image and the cluster allocation.

`python launch_experiment_backbone.py <datasetId> --epochs 40`

## Detector training

`python launch_experiment_detector.py --backbone_experiment_id <backboneId>  <datasetId>`
