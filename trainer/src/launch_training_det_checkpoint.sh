#!/bin/bash
# Train from a EfficientDet D0 checkpoint, it is finetunning
python main.py --mode=train_and_eval \
    --training_file_pattern=tfrecord/*.tfrecord \
    --validation_file_pattern=tfrecord/*.tfrecord \
    --model_name=efficientdet-d0 \
    --model_dir=/tmp/efficientdet-d0-finetune  \
    --ckpt=efficientdet-d0  \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --eval_samples=100 \
    --num_examples_per_epoch=100 \
    --num_epochs=10  \
    --hparams=tfrecord/config.yaml
