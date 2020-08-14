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

You should pass the number of epochs, bear in mind that the learning rate falls
exponentially with the number of epochs. A typical good value for 700K images is
20-30 epochs.
```
# Train for 40 epochs,
python trainer/launch_experiment_backbone.py <experimentId> --epochs 40
# Download the trained model
python trainer/download_experiment_files.py <experimentId>
# Perform inference on an image
python trainer/inference_backbone.py --experiment_id <experimentId> \
    --image_file <image_file>
```

## Detector training

`python trainer/launch_experiment_detector.py --backbone_experiment_id <backboneId>  <experimentId>`
