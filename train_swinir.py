import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model_swinir as model
import loss as Loss
import numpy as np
from torchvision import transforms

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
os.environ['CUDA_VISIBLE_DEVICES']='4'


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def train(config):
	os.environ['CUDA_VISIBLE_DEVICES']='4'

	DarkLighter = model.enhancer().cuda()
	DarkLighter.apply(weights_init)
	if config.load_pretrain == True:
	    DarkLighter.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
	print
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	L_color =Loss.L_color()
	L_cen = Loss.L_cen(16,0.6)
	L_ill = Loss.L_ill()
	L_perc = Loss.perception_loss()
	

	optimizer = torch.optim.Adam(DarkLighter.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	DarkLighter.train()

	for epoch in range(config.num_epochs):
		for iteration, img_lowlight in enumerate(train_loader):

			img_lowlight = img_lowlight.cuda()

			enhanced_image,A  = DarkLighter(img_lowlight)

			Loss_ill = 1600*L_ill(A)
			
			loss_col = 50*torch.mean(L_color(enhanced_image))

			loss_cen = 10*torch.mean(L_cen(enhanced_image))
			
			loss_perc = 0.001*torch.norm(L_perc(enhanced_image) - L_perc(img_lowlight))

			loss =    Loss_ill   +loss_cen +  loss_col + loss_perc
			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(DarkLighter.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Loss at iteration", iteration+1, ":", loss.item())
				
			if ((iteration+1) % config.snapshot_iter) == 0:
				
				torch.save(DarkLighter.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="/data0/dsy/project/ULRE-Net/data/train/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=30)
	parser.add_argument('--train_batch_size', type=int, default=4)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots_swin/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= " ")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	
