#!/bin/sh
PREDICT_PATH=/home/ljp/Desktop/test/graf_mask
PREDICT_FORMAT=.png

TARGET_PATH=/home/ljp/Desktop/test/label
TARGET_FORMAT=.png

IMG_PATH=/home/ljp/Desktop/test/image
IMG_FORMAT=.png

OUT_PATH=/home/ljp/Desktop/test/graf_rst
OUT_FORMAT=.png

METRIC_FILE=metric_file_graf_v0

python metric_f1.py --predict_path $PREDICT_PATH --predict_format $PREDICT_FORMAT --target_path $TARGET_PATH --target_format $TARGET_FORMAT --img_path $IMG_PATH --img_format $IMG_FORMAT --out_path $OUT_PATH --out_format $OUT_FORMAT --metric_file $METRIC_FILE 

