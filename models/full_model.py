import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .__init__ import *
import math
import numpy as np
import cv2
import random


class FullModel(nn.Module):
    def __init__(self, arch, n_classes, mode, energy_thr):
        self.mode = mode
        self.energy_thr = energy_thr
        super(FullModel, self).__init__()
        if arch == 'resnet18':
            self.feature_dim_s1 = 512
            self.feature_dim_s2 = 512
            self.base_s1 = resnet18(pretrained=True)
            self.base_s2 = resnet18(pretrained=True)
        if arch == 'resnet34':
            self.feature_dim_s1 = 512
            self.feature_dim_s2 = 512
            self.base_s1 = resnet34(pretrained=True)
            self.base_s2 = resnet34(pretrained=True)
        elif arch == 'resnet50':
            self.feature_dim_s1 = 2048
            self.feature_dim_s2 = 2048
            self.base_s1 = resnet50(pretrained=True)
            self.base_s2 = resnet50(pretrained=True)
        elif arch == 'resnet101':
            self.feature_dim_s1 = 2048
            self.feature_dim_s2 = 2048
            self.base_s1 = resnet101(pretrained=True)
            self.base_s2 = resnet101(pretrained=True)
        elif arch == 'alexnet':
            self.feature_dim_s1 = 256
            self.feature_dim_s2 = 256
            self.base_s1 = alexnet(pretrained=True)
            self.base_s2 = alexnet(pretrained=True)
        elif arch == 'vgg16':
            self.feature_dim_s1 = 512
            self.feature_dim_s2 = 512
            self.base_s1 = vgg16_bn(pretrained=True)
            self.base_s2 = vgg16_bn(pretrained=True)
        elif arch == 'googlenet':
            self.feature_dim_s1 = 1024
            self.feature_dim_s2 = 1024
            self.base_s1 = googlenet(pretrained=True)
            self.base_s2 = googlenet(pretrained=True)

        self.sigmoid = nn.Sigmoid()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc_s1 = nn.Linear(self.feature_dim_s1, n_classes)
        self.fc_s2 = nn.Linear(self.feature_dim_s2, n_classes)

    def clsmap_norm(self, feat_map):
        feat_b, feat_c, feat_h, feat_w = feat_map.size()
        feat_map = feat_map.view(feat_b, feat_c, -1).permute(0,2,1)
        heat_map =self.fc_s1(feat_map).permute(0,2,1)

        heat_map = heat_map.view(feat_b, -1, feat_h, feat_w)
        return heat_map

    def featmap_norm(self, feat_map):
        feat_map = feat_map.sum(dim=1).unsqueeze(dim=1)
        feat_map = F.upsample(feat_map, size=(25, 25), mode='bilinear', align_corners = True).squeeze(dim=1)
        feat_b, feat_h, feat_w = feat_map.size(0), feat_map.size(1), feat_map.size(2)

        feat_map = feat_map.view(feat_map.size(0), -1)
        feat_map_max, _ = torch.max(feat_map, dim=1)
        feat_map_min, _ = torch.min(feat_map, dim=1)
        feat_map_max = feat_map_max.view(feat_b, 1)
        feat_map_min = feat_map_min.view(feat_b, 1)
        feat_map = (feat_map - feat_map_min) / (feat_map_max - feat_map_min)
        feat_map = feat_map.view(feat_b, 1, feat_h, feat_w)
        return feat_map

    def structured_searching(self, feat_vec):
        feat_b, feat_l = feat_vec.size(0), feat_vec.size(1)
        str = np.zeros(shape=feat_b, dtype=int)
        end = np.zeros(shape=feat_b, dtype=int)
        inf_total = feat_vec.sum(dim=1)

        len_init_thr = 0.5
        for i in range(feat_b):
            info_max = 0
            cen = 0
            # search center and inital h and w
            for j in range(int(feat_l*len_init_thr/2), int(feat_l*(1-len_init_thr/2))):
                enrgy_thr = feat_vec[i, j-int(feat_l*len_init_thr/2):j+int(feat_l*len_init_thr/2)].sum() / inf_total[i]
                if  enrgy_thr >= info_max:
                    info_max = enrgy_thr    
                    cen = j
            str[i] = max(cen-int(feat_l*len_init_thr/2), 0)
            end[i] = min(cen+int(feat_l*len_init_thr/2), feat_l)

            #search final h 
            enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
            # print ('energy: ', enrgy_thr)
            # print ('str, end: ', str, end)
            if enrgy_thr < self.energy_thr:
                # print ('+++: ')
                while enrgy_thr < self.energy_thr:
                    if str[i] == 0:
                        end[i] += 1
                        enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
                    elif end[i] == feat_l - 1:
                        str[i] -= 1
                        enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
                    elif feat_vec[i, str[i]-1] > feat_vec[i, end[i]+1]:
                        str[i] -= 1
                        enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
                    elif feat_vec[i, str[i]-1] < feat_vec[i, end[i]+1]:
                        end[i] += 1
                        enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
                    # print ('energy: ', enrgy_thr)
                    # print ('str, end: ', str, end)
            else:
                # print ('---: ')
                while enrgy_thr > self.energy_thr:
                    if feat_vec[i, str[i]] > feat_vec[i, end[i]]:
                        end[i] -= 1
                        enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
                    else:
                        str[i] += 1
                        enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
                    # print ('energy: ', enrgy_thr)
                    # print ('str, end: ', str, end)
        str = str.astype(float)
        end = end.astype(float)
        str = str / (feat_l*1.0)
        end = end / (feat_l*1.0)

        return str, end

    def bounding_box(self, feat_map, is_training):
        feat_map = feat_map.squeeze(dim=1)
        feat_b = feat_map.size(0)
        feat_vec_h = feat_map.sum(dim=2)
        feat_vec_w = feat_map.sum(dim=1)

        if not is_training:
            h_str, h_end = self.structured_searching(feat_vec_h)
            w_str, w_end = self.structured_searching(feat_vec_w)
        else:
            h_str = np.zeros(shape=feat_b, dtype=float)
            h_end = np.zeros(shape=feat_b, dtype=float)
            w_str = np.zeros(shape=feat_b, dtype=float)
            w_end = np.zeros(shape=feat_b, dtype=float)
            for i in range(feat_b):
                h_str[i] = random.uniform(0, 1-0.5)
                h_end[i] = h_str[i] + 0.5
                w_str[i] = random.uniform(0, 1-0.5)
                w_end[i] = w_str[i] + 0.5

        return [h_str, h_end, w_str, w_end]

    def img_sampling(self, img, h_str, h_end, w_str, w_end):
        img_b, img_c, img_h, img_w = img.size()
        img_sampled = torch.zeros(img_b, img_c, int(img_h/2), int(img_w/2)).cuda()
        h_str = (h_str*img_h).astype(int)
        h_end = (h_end*img_h).astype(int)
        w_str = (w_str*img_w).astype(int)
        w_end = (w_end*img_w).astype(int)
        for i in range(img_b):
            img_sampled_i = img[i, :, h_str[i]:h_end[i], w_str[i]:w_end[i]].unsqueeze(dim=0)
            img_sampled[i, :] = F.upsample(img_sampled_i, size=(int(img_h/2), int(img_w/2)), mode='bilinear', align_corners = True)

        return img_sampled
 
    def get_parameters(self):
        if self.mode == 's1':
            for i in self.base_s2.parameters():
                i.requires_grad = False
            for i in self.fc_s2.parameters():
                i.requires_grad = False
            params = list(self.base_s1.parameters()) + list(self.fc_s1.parameters()) 
        elif self.mode == 's2':
            for i in self.base_s1.parameters():
                i.requires_grad = False
            for i in self.fc_s1.parameters():
                i.requires_grad = False
            params = list(self.base_s2.parameters()) + list(self.fc_s2.parameters())
        return params
 
    def baseline_searching(self, feat_vec):
        feat_b, feat_l = feat_vec.size(0), feat_vec.size(1)

        str = np.zeros(shape=feat_b, dtype=int)
        end = np.zeros(shape=feat_b, dtype=int)

        for i in range(feat_b):
            for j in range(feat_l):
                if feat_vec[i, j] != 0:
                    if str[i]==0 and j>0:
                        str[i] = j-1
                    elif str[i]==0 and j==0:
                        str[i] =0
                    end[i] = j


        str = str.astype(float)
        end = end.astype(float)
        str = str / (feat_l*1.0)
        end = end / (feat_l*1.0)

        return str, end

    def baseline_bounding_box(self, feat_map, is_training):
        feat_map = (feat_map - 0.7) + 1
        feat_map = feat_map.int().float()

        feat_map = feat_map.squeeze(dim=1)
        feat_b = feat_map.size(0)
        feat_vec_h = feat_map.sum(dim=2)
        feat_vec_w = feat_map.sum(dim=1)

        if not is_training:
            h_str, h_end = self.baseline_searching(feat_vec_h)
            w_str, w_end = self.baseline_searching(feat_vec_w)
        else:
            h_str = np.zeros(shape=feat_b, dtype=float)
            h_end = np.zeros(shape=feat_b, dtype=float)
            w_str = np.zeros(shape=feat_b, dtype=float)
            w_end = np.zeros(shape=feat_b, dtype=float)
            for i in range(feat_b):
                h_str[i] = random.uniform(0, 1-0.5)
                h_end[i] = h_str[i] + 0.5
                w_str[i] = random.uniform(0, 1-0.5)
                w_end[i] = w_str[i] + 0.5

        return [h_str, h_end, w_str, w_end]


    def forward(self, img_s1, img_s2, is_training=True):
        img_b = img_s1.size(0)
        heat_map, [h_str, h_end, w_str, w_end] = None, [None, None, None, None]
        if self.mode == 's1':
            feat_map_s1 = self.base_s1.get_features(img_s1)
            logits_s1 = self.fc_s1(self.pooling(feat_map_s1).view(img_s1.size(0), -1))

            logits = logits_s1

        elif self.mode == 's2':
            with torch.no_grad():
                feat_map_s1 = self.base_s1.get_features(img_s1)                
                logits_s1 = self.fc_s1(self.pooling(feat_map_s1).view(img_b, -1))
                heat_map = self.featmap_norm(feat_map_s1)
                h_str, h_end, w_str, w_end = self.bounding_box(heat_map, is_training)
                # h_str, h_end, w_str, w_end = self.baseline_bounding_box(heat_map, is_training)
                img_s2 = self.img_sampling(img_s2, h_str, h_end, w_str, w_end)

            feat_map_s2 = self.base_s2.get_features(img_s2)                
            logits_s2 = self.fc_s2(self.pooling(feat_map_s2).view(img_b, -1))

            if is_training:
                logits = logits_s2
            else:    
                logits = logits_s1 + logits_s2

        return logits, [h_str, h_end, w_str, w_end], heat_map

