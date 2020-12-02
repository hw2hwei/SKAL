import torch
from torch.autograd import Variable
from utils import *
import csv
import os
from pycm import *


def train(epoch, train_loader, net, optim, criterion):
    print('train at epoch {}'.format(epoch))

    net.train()
    losses = AverageMeter()
    accuracies = AverageMeter()

    for i, (imgs_s1, imgs_s2, labels) in enumerate(train_loader):
        imgs_s1 = Variable(imgs_s1.cuda())    
        imgs_s2 = Variable(imgs_s2.cuda())    
        labels = Variable(labels.cuda())
        logits, _, _ = net(imgs_s1, imgs_s2, is_training=True)

        optim.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optim.step()

        acc = accuracy(logits, labels)
        losses.update(loss.item(), logits.size(0))        
        accuracies.update(acc, logits.size(0))

        if (i%50==0 and i!=0) or i+1==len(train_loader):
            print ('Train:   Epoch[{}]:{}/{}   Loss:{:.4f}   Accu:{:.2f}%'.\
                    format(epoch, i, len(train_loader), float(losses.avg), float(accuracies.avg)*100))
       
    return accuracies.avg

