import torch
import torch.utils.data as data
import os
import torchvision.transforms as transforms
from PIL import Image
import random

class MultiScaleRandomCrop(object):
    def __init__(self, scales, size):
        self.scales = scales
        self.crop_size = size

    def __call__(self, img):
        img_size = img.size[0] 
        scale = random.sample(self.scales, 1)[0]
        re_size = int(img_size / scale)
        img = img.resize((re_size, re_size), Image.BILINEAR)
        x1 = random.randint(0, re_size-img_size)
        y1 = random.randint(0, re_size-img_size)
        x2 = x1 + self.crop_size
        y2 = y1 + self.crop_size
        img = img.crop((x1, y1, x2, y2))
        return img


def make_list(root, split_path):
	list_path = os.path.join(root, split_path)
	data_list = []
	class_dict = {}
	f = open(list_path, 'r')
	line = f.readline()
	while line:
		sample ={}
		line = line.strip('\n')
		img_path, label = line.split(' ')

		sample['img_path'] = img_path
		sample['label'] = label
		data_list.append(sample)
		if label not in class_dict.keys(): 	
			class_dict[label] = [img_path]
		else:
			class_dict[label].append(img_path)

		line = f.readline()
	f.close()
	return data_list, class_dict


class datasets(data.Dataset):
	def __init__(self, root, split_path, transform_s1, transform_s2):
		self.root = root
		self.split_path = split_path 
		self.data_list, self.class_dict = make_list(root, split_path)

		self.transform_s1 = transform_s1
		self.transform_s2 = transform_s2

	def __getitem__(self, idx):
		label = int(self.data_list[idx]['label'])
		img_pth = self.data_list[idx]['img_path']
		img_pth = os.path.join(self.root, img_pth)
		img = Image.open(img_pth).convert('RGB')
		img_s1 = self.transform_s1(img)
		img_s2 = self.transform_s2(img)
		return img_s1, img_s2, label

	def __len__(self):
		return len(self.data_list)


def load_datasets(root, train_list, val_list, mode, batch_size, img_size, n_workers):
	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
		)

	train_transform = transforms.Compose([
		transforms.Resize(int(img_size)),
		transforms.CenterCrop(img_size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
		])
	val_transform = transforms.Compose([
		transforms.Resize(img_size),
		transforms.CenterCrop(img_size),
		transforms.ToTensor(),
		normalize,
		])
	transform_s2 = transforms.Compose([
		transforms.Resize(int(img_size*2)),
		transforms.CenterCrop(int(img_size*2)),
		transforms.ToTensor(),
		normalize,
		])

	train_datasets = datasets(root=root, 
							  split_path=train_list, 
							  transform_s1=train_transform, 
							  transform_s2=transform_s2)
	val_datasets = datasets(root=root, 
							split_path=val_list, 
  							  transform_s1=val_transform, 
							  transform_s2=transform_s2)
	train_loader = torch.utils.data.DataLoader(
							dataset=train_datasets,
							batch_size=batch_size,
							shuffle=True,
							num_workers=n_workers,
							drop_last=False)
	val_loader = torch.utils.data.DataLoader(
							dataset=val_datasets,
							batch_size=batch_size,
							shuffle=True,
							num_workers=n_workers,
							drop_last=False)
	return train_loader, val_loader