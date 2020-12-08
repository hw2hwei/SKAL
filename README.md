# SKAL
This is the Pytorch implementation of SKAL for remote sensing scene image classification, which is accepted by TNNLS 2020.

# Requirements
Pytorch 1.7, python 3.8, CUDA 11.0

# Usage
Build the training and validation list:
> cd ./datasets 

> python build_list  --data_dir your_dataset/images  --out_dir your_dataset/splits  --train_ratio 0.5

Train the model seperately at two scales (followed by validation):

> cd ..

> firstly, training global_area: python main.py  --dataset selected_dataset  --arch selected_cnn_arch  --mode s1 

> second, training local_area:   python main.py  --dataset selected_dataset  --arch selected_cnn_arch  --mode s2 

# Results
NWPU_RESISC45

<table border="0.5px" align="left" bordercolor="black" width="80%" height="100px">
    <tr align="center">
        <td>GoogleNet_{global}</td>
        <td>GoogleNet_{global+local}</td>
    </tr>
    <tr align="center">
        <td>89.40</td>
        <td>90.41</td>
    </tr>
</table>
> 


# Citation
If you want to use the code, please cite: 
> title={Looking Closer at the Scene: Multi-Scale Representation Learning for Remote Sensing Image Scene Classification},

> author={Q. Wang, W. Huang, Z. Xiong, and X. Li},

> journal={IEEE Transactions on Neural Networks and Learning Systems},

> year={2020},

> }
