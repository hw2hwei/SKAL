# SKAL
This is the Pytorch implementation of SKAL for remote sensing scene image classification, which is accepted by TNNLS 2020.

# Requirements
Pytorch 1.7, python 3.8, CUDA 11.0

# Usage
Firstly, build the training and validation list:
> cd ./datasets 

> python build_list  --data_dir your_dataset/images  --out_dir your_dataset/splits  --train_ratio custom_ratio(default: 0.5)


Then, train the model seperately at two scales (followed by validation):

> cd ..

> training scale1: python main.py  --dataset selected_dataset  --arch selected_cnn_arch  --mode s1 

> training scale2: python main.py  --dataset selected_dataset  --arch selected_cnn_arch  --mode s2 

# Citation
If you want to use the code, please cite: 
> title={Looking Closer at the Scene: Multi-Scale Representation Learning for Remote Sensing Image Scene Classification},

> author={Q. Wang, W. Huang, Z. Xiong, and X. Li},

> journal={IEEE Transactions on Neural Networks and Learning Systems},

> year={2020},

> }
