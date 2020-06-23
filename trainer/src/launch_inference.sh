#!/bin/bash
python model_inspect.py --runmode=infer --model_name=$MODEL \
  --hparams="image_size=1920x1280" --max_boxes_to_draw=100 --min_score_thresh=0.4 \
  --ckpt_path=$CKPT_PATH --input_image=img.png --output_image_dir=/tmp
