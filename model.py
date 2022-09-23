import torch
import torch.nn as nn
import math
import numpy as np
from torchvision import transforms
import cv2
class enhancer(nn.Module):

	def __init__(self):
		super(enhancer, self).__init__()

		self.relu = nn.ReLU(inplace=True) 

		number_f = 32
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True)
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv8 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
		self.e_conv9 = nn.Conv2d(number_f*2,8,3,1,1,bias=True) 
		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


		
	def forward(self, x):

		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4(x3))
		x5 = self.relu(self.e_conv5(torch.cat([x1,x4],1)))
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		x7 = self.relu(self.e_conv7(torch.cat([x3,x6],1)))
		x8 = self.relu(self.e_conv8(torch.cat([x4,x7],1)))
		
		x_r = torch.tanh(self.e_conv9(torch.cat([x7,x8],1)))

		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 1, dim=1)
		

		x = (x) * (r1+1)
		x = (x) * (r2+1)
		x = (x) * (r3+1)
		enhance_image_1 = (x) * (r4+1)
		x = (enhance_image_1) * (r5+1)
		enhance_image = (x) * (r6+1)
		x = (x) * (r6+1)
		x = (x) * (r7+1) 
		enhance_image = (x) * (r8+1)
		
		r = torch.cat([(r1+1),(r2+1),(r3+1),(r4+1),(r5+1),(r6+1),(r7+1),(r8+1)],1)

		
		return enhance_image,r

