# F1-measure-for-semantic-segmentation
F1 measure for semantic segmentation

## metric_f1.py 参数说明：

* predict_path 算法输出结果存放的目录

* predict_format 算法输出结果的类型，例如im12.png, 其类型为 .png

* target_path 人工标注存放的目录

* target_format 人工标注的类型

* img_path 测试图片存放的目录

* img_format 测试图片的类型

* out_path 输出图片存放的目录

* out_format 输出图片的类型

* metric_file F1-measure的最大值、最小值、均值，以及在所有测试图片上的取值存放的文件

## 示例：evaluation.sh


## 注意事项

1. 输出图片中，模型正确预测为前景的部分为原图片对应部分，模型正确预测为背景的部分为黑色，模型错误预测为前景的部分为红色，模型错误预测为背景的部分为蓝色；
2. 人工标准与算法输出的结果的文件名必须相同，后缀可以不同；
3. 程序输出的日志将存放于当前目录下的log.txt文件中。
