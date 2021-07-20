import os
import torch
import torch.nn as nn
from models.full_model import *
import torchvision.transforms as transforms
from args import args_parser
from val import validation
from datasets import load_datasets
from utils import *
from PIL import Image
import cv2
import numpy as np
from time import time
import pdb
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
os.environ["CUDA_VISIBLE_DEVICES"]="0"

args = args_parser()
best_acc = 0

def make_list(root, dataset, split_path):
    list_path = os.path.join(root, split_path.replace('dataset', dataset))
    data_list = []
    class_dict = {}
    f = open(list_path, 'r')
    line = f.readline()
    while line:
        sample ={}
        line = line.strip('\n')
        img_path, label = line.split(' ')

        sample['img_path'] = root + '/' + img_path
        sample['label'] = label
        data_list.append(sample)
        if label not in class_dict.keys():  
            class_dict[label] = [img_path]
        else:
            class_dict[label].append(img_path)
        line = f.readline()
    f.close()
    return data_list

def cls2label(root, dataset, class_path):
    f = open(os.path.join(root, class_path.replace('dataset', dataset)), 'r')
    line = f.readline()
    cls2label_list ={}   
    while line:
        line = line.strip('\n')
        cls, label = line.split(' ')
        cls2label_list[str(cls)] = str(label)
        line = f.readline()
    return cls2label_list

def label2cls(root, dataset, class_path):
    f = open(os.path.join(root, class_path.replace('dataset', dataset)), 'r')
    line = f.readline()
    label2cls_list ={}   
    while line:
        line = line.strip('\n')
        cls, label = line.split(' ')
        label2cls_list[str(label)] = str(cls)
        line = f.readline()
    return label2cls_list

def scaling(x):
    max, min = np.max(x), np.min(x)
    x = (x - min) / (max - min)
    return x

def show_cam_on_image(h_str,h_end, w_str,w_end, img, label, heat_map, img_save_path):
    img_h, img_w = img.shape[:2]
    heat_map = heat_map.reshape(heat_map.shape[1], heat_map.shape[2], 1)

    feat_map = (heat_map - 0.5) + 1
    feat_map = np.uint8(np.float32(feat_map))
    feat_map = cv2.resize(feat_map, (img_w,img_h))
    feat_map = cv2.applyColorMap(np.uint8(255*feat_map), cv2.COLORMAP_JET)
    feat_map = np.float32(feat_map) 
    cv2.imwrite(img_save_path.replace('.tif', '_heatmap_bin.tif').replace('.jpg', '_heatmap_bin.jpg'), 
                np.uint8(feat_map))

    heat_map = cv2.resize(heat_map, (img_w,img_h))
    heat_map = cv2.applyColorMap(np.uint8(255*heat_map), cv2.COLORMAP_JET)
    heat_map = np.float32(heat_map) 
    cam = heat_map + np.float32(img)
    cam = cam / np.max(cam)

    cv2.imwrite(img_save_path, np.uint8(img))

    img_fusion = img*0.0 + heat_map*1.0
    cv2.imwrite(img_save_path.replace('.tif', '_heatmap.tif').replace('.jpg', '_heatmap.jpg'), 
                np.uint8(img_fusion))


    img_fusion = img*0.5 + heat_map*0.5
    cv2.rectangle(img_fusion, (w_str, h_str), (w_end, h_end), (0, 255, 0), 3, 4) 
    cv2.imwrite(img_save_path.replace('.tif', '_bounding.tif').replace('.jpg', '_bounding.jpg'), 
                np.uint8(img_fusion))


def plot_confusion_matrix(dataset, true_list, pred_list, label2cls_list):
    labels = []
    for key, value in label2cls_list.items():
        labels.append(value)
    tick_marks = np.float32(np.array(range(len(labels)))) + 0.5

    cm = confusion_matrix(true_list, pred_list)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    
    fontsize_axis = 4.2
    fontsize_prop = 2.53
    barsize = 5
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_norm[y_val][x_val]
        if c > 0.01:
            color="white" if c > 0.5 else "black"
            plt.text(x_val, y_val, '%0.2f'%(c,), color=color, fontsize=fontsize_prop, va='center', ha='center')

    plt.gca().set_xticks(tick_marks)
    plt.gca().set_yticks(tick_marks)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-', linewidth=0.3)

    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues) 
    xlocations = np.array(range(len(labels))) 
    plt.xticks(xlocations, labels, fontsize=fontsize_axis, rotation=270) 
    plt.yticks(xlocations, labels, fontsize=fontsize_axis) 

    cb = plt.colorbar(shrink=1.0) 
    cb.ax.tick_params(labelsize=barsize)

    plt.tight_layout()
    plt.savefig('./save_status/confusion_matrix_' + dataset + '.pdf', format='pdf')
    # plt.show()

# attention
if __name__ == '__main__':  
    # bulid model
    if args.dataset=='AID':
        n_classes = 30
    elif args.dataset=='UCM':
        n_classes = 21
    elif args.dataset=='NWPU-RESISC45':
        n_classes = 45
    elif args.dataset=='RSSCN7':
        n_classes = 7
    net = FullModel(arch=args.arch,  
                    n_classes=n_classes, 
                    mode=args.mode,
                    energy_thr=args.energy_thr).cuda()
    resume_path = args.resume_path.replace('dataset', args.dataset)   \
                                  .replace('arch', args.arch)  \
                                  .replace('mode', str(args.mode))
    if os.path.exists(resume_path):
        resume = torch.load(resume_path)
        start_epoch = resume['epoch']
        net.load_state_dict(resume['state_dict'], strict=False)
        print ('Load checkpoint {}'.format(resume_path))    
    net = net.cuda().eval()
    criterion = nn.CrossEntropyLoss().cuda()

    cls2label_list = cls2label(args.data_dir, args.dataset, args.class_list)
    label2cls_list = label2cls(args.data_dir, args.dataset, args.class_list)
    data_list = make_list(args.data_dir, args.dataset, args.val_list)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    transform_s1 = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        normalize,
        ])
    transform_s2 = transforms.Compose([
        transforms.Resize(int(args.img_size*2)),
        transforms.CenterCrop(int(args.img_size*2)),
        transforms.ToTensor(),
        normalize,
        ])

    # record loss
    losses = AverageMeter()
    accuracies = AverageMeter()
    # random.shuffle(data_list)

    cnt_total = 0
    pred_list = []
    label_list = []

    beg_time = time()
    for data in data_list:
        cnt_total += 1
        img_pth = data['img_path']
        label = int(data['label'])
        print (cnt_total, ': ', img_pth)   

        img_save_dir = './attvisual_image'
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        img = Image.open(img_pth).convert('RGB')
        img_tensor_s1 = transform_s1(img)
        img_tensor_s2 = transform_s2(img)
        img_tensor_s1 = img_tensor_s1.view(1, img_tensor_s1.size(0), img_tensor_s1.size(1), img_tensor_s1.size(2)).cuda()
        img_tensor_s2 = img_tensor_s2.view(1, img_tensor_s2.size(0), img_tensor_s2.size(1), img_tensor_s2.size(2)).cuda()
        pred, [h_str, h_end, w_str, w_end], heat_map = net(img_tensor_s1, img_tensor_s2, is_training=False)
        _, pred = torch.max(pred, dim=1)
        pred = int(pred.cpu().detach().numpy())
        pred_list.append(pred)
        label_list.append(label)

        heat_map = heat_map[0].cpu().detach().numpy()
        h_str = h_str[0]
        h_end = h_end[0]
        w_str = w_str[0]
        w_end = w_end[0]

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img_h, img_w = img.shape[:2]
        h_str = (h_str*img_h).astype(np.int)
        h_end = (h_end*img_h).astype(np.int)
        w_str = (w_str*img_w).astype(np.int)
        w_end = (w_end*img_w).astype(np.int)
        img_save_path = img_save_dir + '/' + img_pth.split('/')[5]

#         if cnt_total > 300:
#             break        

        # img
        if pred != label:
            img_save_path = img_save_path.replace('.jpg', '_wrong_{}.jpg'.format(label2cls_list[str(pred)]))  \
                                         .replace('attvisual_image/', 'attvisual_image/wrong/')
            show_cam_on_image(h_str,h_end,w_str,w_end, img, pred, heat_map, img_save_path)
        else:        
            img_save_path = img_save_path.replace('attvisual_image/', 'attvisual_image/true/')
            show_cam_on_image(h_str,h_end,w_str,w_end, img, pred, heat_map, img_save_path)
    end_time = time()
    print ("time: ", end_time - beg_time)

    plot_confusion_matrix(args.dataset, label_list, pred_list, label2cls_list)

