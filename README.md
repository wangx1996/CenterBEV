# CenterBEV
A deep learning method for pointcloud object detection.

[![torch 1.3](https://img.shields.io/badge/torch-1.3-red.svg)](https://github.com/wangx1996/CenterBEV)  [![python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://github.com/wangx1996/CenterBEV)

### Introdcution

This is an anchor free method for pointcloud object detection by using bird eye view.

This project is not finished yet, it has a lot of parts to be improved. 

If you are intreseted in this project, you can try to change the code and make this work better.

### Structure
![Image text](https://github.com/wangx1996/CenterBEV/blob/main/structure/structure.png)

### 1.Clone Code

    git clone https://github.com/wangx1996/CenterBEV.git CenterBEV
    cd CenterBEV/

### 2.Install Dependence
#### 2.1 base pacakge
    pip install -r requirements.txt
    
for anaconda

    conda install scikit-image scipy numba pillow matplotlib
    pip install fire tensorboardX protobuf opencv-python
    
#### 2.2 DCN

Please download DCNV2 from [https://github.com/jinfagang/DCNv2_latest](https://github.com/jinfagang/DCNv2_latest) to fit torch 1.

Put the file into 

    ./src/model/
    
then 

    ./make.sh

#### 2.3 Setup cuda for numba

    export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
    export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
    export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
    
    
### 3. Prepaer data

KITTI dataset

You can Download the KITTI 3D object detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

It includes:
Velodyne point clouds (29 GB)

Training labels of object data set (5 MB)

Camera calibration matrices of object data set (16 MB)

Left color images of object data set (12 GB) 

Data structure like

    └── KITTI_DATASET_ROOT
       ├── classes_names.txt    
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   └── velodyne
       └── testing     <-- 7580 test data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   └── velodyne
       └── ImageSets
           ├── train.txt
           ├── val.txt
           ├── trainval.txt
           └── test.txt
           
           
           
### 4. How to Use

First, make sure the dataset dir is right in your train.py file

Then run

    python train.py --gpu_idx 0 --arch DLA_34 --saved_fn dla34 --batch_size 4
    
Tensorboard
    
    cd logs/<saved_fn>/tensorboard/
    tensorboard --logdir=./
    
    
if you want to test the work

    python test.py --gpu_idx 0 --arch DLA_34 --pretrained_path ../checkpoints/**/** --peak_thresh 0.4
    
if you want to evaluate the work

    python evaluate.py --gpu_idx 0 --arch DLA_34 --pretrained_path ../checkpoints/**/**
    
    
### Reference

Thanks for all the great works.

[1] [SFA3D](https://github.com/maudzung/SFA3D)
[2] [Complex_Yolo](https://github.com/maudzung/Complex-YOLOv4-Pytorch)
[3] [CenterNet: Objects as Points](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1904.07850), [[PyTorch Implementation](https://github.com/xingyizhou/CenterNet)]
[4] [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211) [[final version code](https://github.com/jinfagang/DCNv2_latest)]


### Result


![Image text](https://github.com/wangx1996/CenterBEV/tree/main/result/2.png)
![Image text](https://github.com/wangx1996/CenterBEV/tree/main/result/8.png)
![Image text](https://github.com/wangx1996/CenterBEV/tree/main/result/11.png)
