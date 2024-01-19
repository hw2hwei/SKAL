import os
import random
import argparse

parser = argparse.ArgumentParser(description='Build the splits of remote datasets')
parser.add_argument('--root', default='.', type=str)
parser.add_argument('--data_dir', default='./RSSCN7/images', type=str)
parser.add_argument('--out_dir', default='./RSSCN7/splits', type=str)
parser.add_argument('--train_ratio', default=0.50, type=float)


def build_classInd(root, data_dir, out_dir):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	data_path = os.path.join(root, data_dir)
	classInd_path = os.path.join(root, out_dir, 'classInd.txt')

	classInd = {}
	num = 0
	for f, dirs, fs in os.walk(data_path):
		for cls in dirs:
			if cls != []:
				classInd[cls] = str(num)
				num += 1

	f = open(classInd_path, 'w')
	for k, v in classInd.items():
		new_context = str(k) + ' ' + v + '\n'
		f.write(new_context)
	f.close()
	return classInd


def build_list(root, data_dir, out_dir, train_ratio):
	# build classInd
	classInd = build_classInd(root, data_dir, out_dir)
	print (classInd)

	# build train and val list
	train_list = []
	val_list = []
	for dir, dirs, fs in os.walk(data_dir):
		if fs != []:
			train_num = int(train_ratio*len(fs))
			random.shuffle(fs)

			files = []
			for f in fs:
				cls = dir.replace(data_dir, '').strip('\\').strip('/')
				print (dir)
				files.append(os.path.join(dir, f + ' ' + classInd[cls]))

			train_list_each = files[0:train_num]
			val_list_each = files[train_num:]
			train_list.extend(train_list_each)
			val_list.extend(val_list_each)

	train_path = os.path.join(root, out_dir, 'train_split.txt')
	val_path = os.path.join(root, out_dir, 'val_split.txt')

	f = open(train_path, 'w')
	for img_path in train_list:
		new_context =img_path + '\n'
		f.write(new_context)
	f.close()
	f = open(val_path, 'w')
	for img_path in val_list:
		new_context =img_path + '\n'
		f.write(new_context)
	f.close()


if __name__ == '__main__':
	global args
	args = parser.parse_args()

	root = args.root	
	data_dir = args.data_dir
	train_ratio = args.train_ratio
	out_dir = args.out_dir

	build_list(root, data_dir, out_dir, train_ratio)
