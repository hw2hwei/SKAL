# SKAL
This is the Pytorch implementation of SKAL for remote sensing scene image classification, which is accepted by TNNLS 2020.

# Requirements
Pytorch 1.7, python 3.8, CUDA 11.0  

# Usage
Dataset would be arranged as the following form (taking AID as an example):
> ./datasets  
> ./datasets/AID  
> ./datasets/AID/splits  
> ./datasets/AID/images  
  ./datasets/AID/images/agricultural  
  ./datasets/AID/images/airplane  
  ./datasets/AID/images/baseballdiamond  


  
Build the training and validation list (Taking AID as an example):
> cd ./datasets  
> python build_list  --data_dir AID/images  --out_dir AID/splits  --train_ratio 0.5  

Train the model at two seperate scales (validate the model every epoch):  
> cd ..  
> firstly, training global_area: python main.py  --dataset selected_dataset  --arch selected_cnn_arch  --mode s1  
> secondly, training local_area: python main.py  --dataset selected_dataset  --arch selected_cnn_arch  --mode s2  

# Results
NWPU_RESISC45  

|  Method                   | 10\% for training | 20\% for training |
|     ----                  |      ----         |      ----         |
| GoogleNet_{global}        |      77.44        |      83.69        |
| GoogleNet_{global+local}  |      80.28        |      85.34        |
| ResNet18_{global}         |      88.91        |      91.77        |
| ResNet18_{global+local}   |      90.04        |      92.79        |
| GoogleNet_{global}        |      89.40        |      91.93        |
| GoogleNet_{global+local}  |      90.41        |      92.95        |



# Citation
If you want to use the code, please cite: 
> title={Looking Closer at the Scene: Multi-Scale Representation Learning for Remote Sensing Image Scene Classification},  
> author={Q. Wang, W. Huang, Z. Xiong, and X. Li},  
> journal={IEEE Transactions on Neural Networks and Learning Systems},  
> year={2020},  
> }
