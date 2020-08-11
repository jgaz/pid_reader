# EfficientDet detector

Check out


## Installation

Follow python installation from [Tensorflow 2 Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)


## Training and evaluation

[TF2 Training](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md)


## Experiments

### 700K 500x500 6 objects per diagram,
- Backbone: 90% accuracy, 700K, 500 classes: P-Process,L-Piping,J-Instrument

`python launch_experiment_backbone.py 8c3e8163488f103dc7aef34af6ff74891b1fe8c7 --epochs 40`

- Detector:
`python launch_experiment_detector.py --backbone_experiment_id 8c3e8163488f103dc7aef34af6ff74891b1fe8c7  20b110c8abd721c309576ab13994a32fadc09a28`
