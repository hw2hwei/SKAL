import argparse

def args_parser():
	parser = argparse.ArgumentParser(description='Build the splits of remote datasets')
	
	# root setting
	parser.add_argument('--data_dir', default='/home/huangwei/Datasets/Remote_Sensing_Scene_Classification', type=str)
	parser.add_argument('--dataset', default='UCM', type=str,
						 			 choices=['AID', 'UCM', 'OPTIMAL31', 'WHU', 'NWPU-RESISC45', 'RSSCN7'])
	parser.add_argument('--class_list', default='dataset/splits/classInd.txt', type=str)
	parser.add_argument('--train_list', default='dataset/splits/train_split.txt', type=str)
	parser.add_argument('--val_list', default='dataset/splits/val_split.txt', type=str)
	parser.add_argument('--resume_path', default='checkpoints/dataset_arch.pth', type=str)

	# model alexnet
	parser.add_argument('--arch', default='resnet18', type=str,
									choices=['alexnet', 'resnet18', 'resnet18', 'vgg16'])
	parser.add_argument('--mode', default='s1', type=str, 
									choices=['s1', 's2'])
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--img_size', default=224, type=int)
	parser.add_argument('--energy_thr', default=0.7, type=float)	
	parser.add_argument('--n_workers', default=8, type=int)

	# train setting
	parser.add_argument('--start_epoch', default=0, type=int)
	parser.add_argument('--epochs', default=50, type=int)
	parser.add_argument('--step_size', default=20, type=int)
	parser.add_argument('--lr', default=1e-4, type=float)	
	args = parser.parse_args()
	return args

