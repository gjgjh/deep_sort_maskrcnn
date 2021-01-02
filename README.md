# MaskRCNN + Deep Sort

SLAM和深度学习结合是一个发展趋势，目前已有许多这方面的研究工作。通常目标检测/语义分割/实例分割用来提供语义观测，目标追踪用来提供数据关联，从而应用于一个SLAM系统中。本仓库实现语义SLAM中的关键部分，即实例分割和目标追踪功能，代码主要来自[maskrcnn-benchmark](https://github.com/umbertogriffo/maskrcnn-benchmark) 和 [deep_sort](https://github.com/nwojke/deep_sort)。仅为个人学习使用，仍有许多地方需要完善。更详细的安装和使用说明请参考maskrcnn和deep sort原仓库。

## 依赖与安装

### 依赖

本仓库依赖以下第三方库，并只在以下特定版本上进行了测试。

- PyTorch 1.2.0
- torchvision 0.4.0
- cocoapi 2.0
- yacs 0.1.8
- matplotlib 3.3.3
- GCC >= 4.9
- OpenCV 4.4.0
- Sklearn 0.22.2

### 安装

```bash
conda create -n maskrcnn_benchmark python=3.6
conda activate maskrcnn_benchmark

pip install ninja yacs cython matplotlib tqdm opencv-python scikit-learn==0.22.2

# install pytorch
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

mkdir maskrcnn && cd maskrcnn
export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
python setup.py install --cuda_ext --cpp_ext

# install deep_sort_maskrcnn
cd $INSTALL_DIR
git clone https://github.com/gjgjh/deep_sort_maskrcnn.git
cd deep_sort_maskrcnn/
python setup.py build develop

unset INSTALL_DIR
```

## 运行

```bash
cd deep_sort_maskrcnn/demo

# by default, it doesn't enable the tracker. For best results, use min-image-size 800
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml" --confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cuda MODEL.MASK_ON True 

# can also run it on the CPU
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml" --confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cpu MODEL.MASK_ON True 

# enable the tracker
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml" --confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cuda MODEL.MASK_ON True TRACKER.ENABLE True 

# enable the tracker and save tracked objects's images to relative folders
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml" --confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cuda MODEL.MASK_ON True TRACKER.ENABLE True TRACKER.EXTRACT_FROM_MASK.ENABLE True

# enable the tracker and save tracked objects's images to relative folders with transparent background
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml" --confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cuda MODEL.MASK_ON True TRACKER.ENABLE True TRACKER.EXTRACT_FROM_MASK.ENABLE True TRACKER.EXTRACT_FROM_MASK.TRANSPARENT True 

# can also resize the images to a specific size
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml" --confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cuda MODEL.MASK_ON True TRACKER.ENABLE True TRACKER.EXTRACT_FROM_MASK.ENABLE True TRACKER.EXTRACT_FROM_MASK.TRANSPARENT True  TRACKER.EXTRACT_FROM_MASK.DSIZE 800
```

运行后，结果默认保存在demo文件夹下：

- `extracted_object`：分开保存了每个跟踪到的object的图像

- `visualize`：保存了用于可视化的每帧图像

- `semantic`：保存了跟踪和实例分割的结果。示例格式如下：

  ```
  num_objects
  objectID1 label1 confidence1
  objectID2 label2 confidence2
  ...
  rows cols
  0 0 0 0 0 ... 0 objectID1 0
  objectID2 0 0 0 0 ... 0 objectID1 0
  objectID2 0 0 0 0 ... 0 0 0
  0 0 0 0 0 ... 0 0 0
  ```

  第一行为当前帧**检测并跟踪**到的object数量，下面num_objects行分别为每一个object的ID、类别和置信度。接下来一行为图像的行列号。最后是rows x cols大小的实例分割图像。其中0表示背景或被忽略的结果，非0值为object的ID，用来表示object实例分割结果。

## 示例结果

<img src="./demo/tracking.gif" width = 40% height = 40% />

