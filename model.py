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
		x0 = self.upsample(x)
		x0 = self.upsample(x0)
		x1 = self.relu(self.e_conv1(x0))
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
		x_r = self.maxpool(x_r)
		x_r = self.maxpool(x_r)
		'''
		######################add##############################
		y0 = self.upsample(x)
		y1 = self.relu(self.e_conv1(y0))
		y2 = self.relu(self.e_conv2(y1))
		y3 = self.relu(self.e_conv3(y2))
		y4 = self.relu(self.e_conv4(y3))
		y5 = self.relu(self.e_conv5(torch.cat([y1,y4],1)))
		y6 = self.relu(self.e_conv6(torch.cat([y2,y5],1)))
		y7 = self.relu(self.e_conv7(torch.cat([y3,y6],1)))
		y8 = self.relu(self.e_conv8(torch.cat([y4,y7],1)))
		y_r = torch.tanh(self.e_conv9(torch.cat([y7,y8],1)))
		y_r = self.maxpool(y_r)

		z1 = self.relu(self.e_conv1(x))
		z2 = self.relu(self.e_conv2(z1))
		z3 = self.relu(self.e_conv3(z2))
		z4 = self.relu(self.e_conv4(z3))
		z5 = self.relu(self.e_conv5(torch.cat([z1,z4],1)))
		z6 = self.relu(self.e_conv6(torch.cat([z2,z5],1)))
		z7 = self.relu(self.e_conv7(torch.cat([z3,z6],1)))
		z8 = self.relu(self.e_conv8(torch.cat([z4,z7],1)))
		z_r = torch.tanh(self.e_conv9(torch.cat([z7,z8],1)))

		x_r = (0.6*x_r) + (0.3*y_r) + (0.1*z_r)
        ########################################################
		'''
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

