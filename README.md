# SKAL
This is the Pytorch implementation of SKAL for remote sensing scene image classification, which is accepted by TNNLS 2020.

# Requirements
Pytorch 1.7, python 3.8, CUDA 11.0

# Usage
training (followed by validation):

training scale1: python main.py --dataset selected_dataset --arch selected_cnn_arch --mode s1 

training scale2: python main.py --dataset selected_dataset --arch selected_cnn_arch --mode s2 

# Citation
title={Looking Closer at the Scene: Multi-Scale Representation Learning for Remote Sensing Image Scene Classification},
author={Q. Wang, W. Huang, Z. Xiong, and X. Li},
journal={IEEE Transactions on Neural Networks and Learning Systems},
year={2020},
}
