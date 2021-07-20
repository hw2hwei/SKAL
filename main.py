import torch
import torch.nn as nn
import os
from args import args_parser
from train import train
from val import validation
from datasets import load_datasets
from collections import OrderedDict
from models.full_model import *
import pdb
from time import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"
args = args_parser()
best_acc = 0


if __name__ == '__main__':	
    # load datasets feat
    train_list = args.train_list.replace('dataset', args.dataset)
    val_list = args.val_list.replace('dataset', args.dataset)
    train_loader, val_loader = load_datasets(args.data_dir, 
                                             train_list, 
                                             val_list, 
                                             args.mode,  
                                             args.batch_size, 
                                             args.img_size, 
                                             args.n_workers)

    # bulid model
    resume_path = args.resume_path.replace('dataset', args.dataset)  \
                                  .replace('arch', args.arch)   
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
    if os.path.exists(resume_path):
        resume = torch.load(
        )
        net.load_state_dict(resume['state_dict'], strict=False)
        print ('Load checkpoint {}'.format(resume_path))

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optim = torch.optim.Adam(net.get_parameters(), lr=args.lr)
    sche = torch.optim.lr_scheduler.StepLR(optim, step_size=args.step_size)

    all_time = 0
    best_acc, val_acc = validation(0, best_acc, val_loader, net, resume_path, criterion)
    # pdb.set_trace()
    file_name = '{}_{}.txt'.format(args.dataset, args.mode)

    for i in range(args.start_epoch, args.epochs):
        beg_time = time()
        train_acc = train(i, train_loader, net, optim, criterion)
        end_time = time()
        all_time = all_time + (end_time - beg_time)
        print ('training_time: ', all_time)

        best_acc, val_acc = validation(i, best_acc, val_loader, net, resume_path, criterion)
        with open(file_name, 'a') as file:
            file.write(str(i) + ' ' + str(val_acc) + ' ' + '\n')

        sche.step()


 
