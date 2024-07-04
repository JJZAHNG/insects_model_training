# 昆虫种类识别模型训练

## 依赖

```python
pip3 install tensorflow opencv-python
```

## 运行

```python
python3 train.py
```

## 项目目录

```kotlin
insects/
├── data/
│   ├── train/
│   ├── validation/
├── models/
│   ├── insect_classifier_model.h5
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   └── insects.names
├── train.py
└── yolo_infer.py
```


## WEIGHTS
下载YOLOv3文件
下载权重文件 yolov3.weights
https://pjreddie.com/media/files/yolov3.weights
你可以从以下链接下载： yolov3.weights​ (PJREDDIE)​

## CFG
下载配置文件 yolov3.cfg
https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
你可以从以下链接下载： yolov3.cfg

## Names
下载类别名称文件 coco.names(替换成自己的insects.names)
然后写进自己已训练的昆虫种类
你可以从以下链接下载： coco.names
https://github.com/pjreddie/darknet/blob/master/data/coco.names
