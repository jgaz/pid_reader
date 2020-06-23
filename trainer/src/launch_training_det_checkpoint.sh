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
    --eval_samples=10 \
    --num_examples_per_epoch=10 \
    --num_epochs=6  \
    --hparams=tfrecord/config_finetune.yaml
