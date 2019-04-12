# -*- coding: utf-8 -*-
import numpy as np
import glob
import argparse
import os
import re
import sys
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)

parser = argparse.ArgumentParser(description='F1 measure for salient object extraction')
parser.add_argument('--predict_path', type = str, default = "", help = 'mask outputting by your algorithm')
parser.add_argument('--predict_format', type = str, default = "", help = 'format of your mask')
parser.add_argument('--target_path', type = str, default = "", help = 'groundtruth')
parser.add_argument('--target_format', type = str, default = "", help = "format of groundtruth")
parser.add_argument('--img_path', type = str, default = "", help = "original image")
parser.add_argument('--img_format', type = str, default = "", help = "format of original image")
parser.add_argument('--out_path', type = str, default = "", help = "result path")
parser.add_argument('--out_format', type = str, default = "", help = "format of result")
parser.add_argument('--metric_file', type = str, default = "", help = "f1 measure etc.")

args = parser.parse_args()

predict_path = args.predict_path
predict_format = args.predict_format
target_path = args.target_path
target_format = args.target_format
img_path = args.img_path
img_format = args.img_format
out_path = args.out_path

if not os.path.exists(out_path):
    os.makedirs(out_path)

out_format = args.out_format
metric_file = args.metric_file

logger.info('Evaluation of '+metric_file)

predict_pattern = predict_path + '/*' + predict_format
predict_paths = glob.glob(predict_pattern)

target_pattern = target_path + '/*' + target_format
target_paths = glob.glob(target_pattern)

predict_file_num = len(predict_paths)
target_file_num = len(target_paths)

# 图片数量必须相等
if predict_file_num != target_file_num:
    logger.info('错误！输出图片的数量与ground truth的数量不相等！')
    sys.exit(0)

file_num = predict_file_num
logger.info('评价图片数量： '+str(file_num))

Precision_s = [] 
Recall_s = [] 
F1_measure_s = [] 
F1_measure_dict = {}
aes = []

# 计数变量
i = 0

# 由于predict_mask与target_mask的存储格式为png，灰度值为0,1,2,...,255，因此需要设定一个阈值进行二值化
thres = 100

for file_path in predict_paths:
    logger.info(file_path)
    file_name_with_ext = os.path.basename(file_path)
    file_name = os.path.splitext(file_name_with_ext)[0]

    target_file_path = os.path.join(target_path, file_name + target_format)
    if not os.path.isfile(target_file_path):
        logger.info('错误！文件'+target_file_path+'不存在！')
        sys.exit(0)

    img_file_path = os.path.join(img_path, file_name + img_format)
    if not os.path.isfile(img_file_path):
        logger.info('错误！文件'+img_file_path+'不存在！')
        sys.exit(0)

    predict_mask = np.array((Image.open(file_path)).convert('L'))
    height, width = predict_mask.shape
    if not np.max(predict_mask) == 255:
        logger.info('错误！目标掩模的最大值必须为255！')
        sys.exit(0)

    predict_mask[predict_mask < thres] = 0
    predict_mask[predict_mask >= thres] = 1

    target_mask = Image.open(target_file_path)
    height_1, width_1 = target_mask.size
    if not (height_1 == height and width_1  == width):
        target_mask = target_mask.resize((width, height), Image.ANTIALIAS)

    if not np.max(target_mask) == 255:
        logger.info('错误！人工标注结果的最大值必须为255！')
        sys.exit(0)

    target_mask = np.array(target_mask.convert('L'))
    target_mask[target_mask < thres] = 0
    target_mask[target_mask >= thres] = 1
    
    # 计算单张图像的误差
    aes.append(np.sum(predict_mask != target_mask) * 1.0 / ( height * width ))

    img = Image.open(img_file_path)
    height_2, width_2 = img.size
    if not (height_2 == height and width_2 == width):
        img = img.resize((width, height), Image.ANTIALIAS)

    img_out = np.array(img)
    img_out = img_out[:,:,:3]
    img_out[predict_mask == 0, :] = (0, 0, 0)

    accum = predict_mask + target_mask
    TP = np.sum(accum == 2)

    diff1 = predict_mask - target_mask
    FP = np.sum(diff1 == 1)
    img_out[diff1 == 1] = (255, 0, 0)

    diff2 = target_mask - predict_mask
    FN = np.sum(diff2 == 1)
    img_out[diff2 == 1] = (0, 0, 255)

    plt.imsave(os.path.join(out_path, file_name + out_format), img_out)

    if ( TP + FP ) == 0 or ( TP + FN ) == 0:
        logger.info('错误！不能计算查准率和查全率！')
        sys.exit(0)

    Precision = TP*1.0 / ( TP + FP )
    Recall = TP*1.0 / ( TP + FN )

    if ( Precision + Recall ) == 0:
        F1_measure = 0
    else:
        F1_measure = 2 * Precision * Recall / ( Precision + Recall )

    Precision_s.append(Precision)
    Recall_s.append(Recall)
    F1_measure_s.append(F1_measure)
    F1_measure_dict[file_name_with_ext] = F1_measure

    i = i + 1
try:
    Recall_s = np.array(Recall_s)
    Precision_s = np.array(Precision_s)
    F1_measure_s = np.array(F1_measure_s)
    aes = np.array(aes)
    metric_file_ = open(metric_file, 'w')
    metric_file_.write("min F1_measure: " + str(F1_measure_s.min()) + "\n")
    metric_file_.write("max F1_measure: " + str(F1_measure_s.max()) + "\n")
    metric_file_.write("mean F1_measure: " + str(F1_measure_s.mean()) + "\n")
    metric_file_.write("std F1_measure: " + str(F1_measure_s.std()) + "\n")
    metric_file_.write("min Recall: " + str(Recall_s.min()) + "\n")
    metric_file_.write("max Recall: " + str(Recall_s.max()) + "\n")
    metric_file_.write("mean Recall: " + str(Recall_s.mean()) + "\n")
    metric_file_.write("std Recall: " + str(Recall_s.std()) + "\n")
    metric_file_.write("min Precision: " + str(Precision_s.min()) + "\n")
    metric_file_.write("max Precision: " + str(Precision_s.max()) + "\n")
    metric_file_.write("mean Precision: " + str(Precision_s.mean()) + "\n")
    metric_file_.write("std Precision: " + str(Precision_s.std()) + "\n")
    metric_file_.write('MAE: ' + str(aes.mean()) + "\n")
    metric_file_.write(" ".join(str(v) + ':' + str(F1_measure_dict[v]) for v in F1_measure_dict))
finally:
    if metric_file_:
        metric_file_.close()
