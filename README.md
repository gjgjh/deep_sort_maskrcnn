# Maskrcnn + Deep Sort

本仓库实现实例分割和目标追踪功能，代码主要来自[maskrcnn-benchmark](https://github.com/umbertogriffo/maskrcnn-benchmark) 和 [deep_sort](https://github.com/nwojke/deep_sort)。仅为个人学习使用，仍有许多地方需要完善。

> 更详细的安装和使用说明请参考原仓库

## 依赖与安装

### 依赖

本仓库依赖以下第三方库，并只在以下特定版本上进行了测试。

- PyTorch 1.2.0
- torchvision 0.4.0
- cocoapi 
- yacs 0.1.8
- matplotlib 3.3.3
- GCC >= 4.9
- OpenCV 4.4.0
- Sklearn

### 安装

```bash
conda create -n maskrcnn_benchmark python=3.6
conda activate maskrcnn_benchmark

pip install ninja yacs cython matplotlib tqdm opencv-python scikit-learn

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

# by default, it doesn't enable the tracker
# for best results, use min-image-size 800
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
--confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cuda MODEL.MASK_ON True 
# can also run it on the CPU
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
--confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cpu MODEL.MASK_ON True 
# or enable the tracker
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
--confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cpu MODEL.MASK_ON True TRACKER.ENABLE True 
# or enable the tracker and save tracked objects's images to relative folders
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
--confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cpu MODEL.MASK_ON True TRACKER.ENABLE True TRACKER.EXTRACT_FROM_MASK.ENABLE True
# or enable the tracker and save tracked objects's images to relative folders with transparent background
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
--confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cpu MODEL.MASK_ON True TRACKER.ENABLE True TRACKER.EXTRACT_FROM_MASK.ENABLE True TRACKER.EXTRACT_FROM_MASK.TRANSPARENT True 
# or also resize the images to a specific size
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
--confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cpu MODEL.MASK_ON True TRACKER.ENABLE True TRACKER.EXTRACT_FROM_MASK.ENABLE True TRACKER.EXTRACT_FROM_MASK.TRANSPARENT True  TRACKER.EXTRACT_FROM_MASK.DSIZE 800
```

## 示例结果

TODO