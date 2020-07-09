#!/bin/bash
# Traing using a backbone checkpoint
python main.py --mode=train_and_eval \
    --training_file_pattern=tfrecord/{file_pattern} \
    --validation_file_pattern=tfrecord/{file_pattern} \
    --model_name={MODEL} \
    --model_dir=/tmp/model_dir/{MODEL}-scratch  \
    --backbone_ckpt={backbone_name} \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --eval_samples=100  \
    --num_examples_per_epoch=100 \
    --num_epochs=5  \
    --hparams="num_classes=20,moving_average_decay=0,mixed_precision=true"
