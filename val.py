import torch
from torch.autograd import Variable
from utils import *
from pycm import *
import numpy as np
import torch.nn.functional as F


def validation(epoch, best_acc, val_loader, net, resume_path, criterion):
	print('val at epoch {}'.format(epoch))
	net.eval()
	losses = AverageMeter()
	accuracies = AverageMeter()

	with torch.no_grad():
		for i, (imgs_s1, imgs_s2, labels) in enumerate(val_loader):
			with torch.no_grad():
				img_s1 = Variable(imgs_s1.cuda())    
				img_s2 = Variable(imgs_s2.cuda())    
				labels = Variable(labels.cuda())
				logits, _, _ = net(img_s1, img_s2, is_training=False)
				loss = criterion(logits, labels)
				acc = accuracy(logits, labels)
			losses.update(loss.item(), logits.size(0))
			accuracies.update(acc, logits.size(0))

			if (i%50==0 and i!=0) or i+1==len(val_loader):
				print ('Validation:   Epoch[{}]:{}/{}    Loss:{:.4f}   Accu:{:.2f}%'.   \
						format(epoch, i, len(val_loader), float(losses.avg), float(accuracies.avg)*100))

	print ('best_acc: {:.2f}%'.format(best_acc*100))
	print ('curr_acc: {:.2f}%'.format(accuracies.avg*100))
	if accuracies.avg >= best_acc:
		best_acc = accuracies.avg
		save_file_path = resume_path
		states = {'state_dict': net.state_dict(),
				  'epoch':epoch,
				  'acc':best_acc}
		torch.save(states, save_file_path)
		print ('Saved!')
	print ('')		
	return best_acc, accuracies.avg
