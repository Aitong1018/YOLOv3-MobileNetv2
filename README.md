# Mobilenetv2-yolov3-model

## Environment

Python3.7, Pytorch 1.4


## Datasets perparing

1. Download VOC datasets，getting the compressed file(including VOC2007,VOC2012 and VOC2007TEST).
2. Release the compressed file to "dataVOC/VOCROOT"  directory.
3. run converter.py in "dataVOC"  directory ，generating the "images","labels" file and train.txt、valid.txt. 

or
DOTA数据集
1.在DOTA文件夹下建DOTAROOT文件夹；
2.在DOTAROOT文件夹下将训练集和验证集放在DOTA_train和DOTA_val文件夹下；
3.运行 converter.py文件夹，得到 "images","labels" file and train.txt、valid.txt. 

## The introduction of algrithm

1. training

    ```bash
    python train.py --model_def config/yolov3-voc-mobilenet2.cfg
    ```

2. testing

   ```bash
   python test.py 
   ```
检测图片的输入DOTA/samples
结果输出 output/samples
