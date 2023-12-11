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
import model as model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import pdb
import cv2
def lowlight(image_path, DarkLighter):
	os.environ['CUDA_VISIBLE_DEVICES']='5'
	data_lowlight = Image.open(image_path)
	####add####
	#data_lowlight = data_lowlight.resize((512, 512), Image.ANTIALIAS)
	###########
	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)


	start = time.time()
	enhanced_image,_ = DarkLighter(data_lowlight)
	end_time = (time.time() - start)
	#print(end_time)
	image_path = image_path.split("/")[-1]
	result_path = '/data0/dsy/project/Enhancement_Detection_for_Aerial_Data/object_detection/datasets/dark_face/val_light/images/'
	torchvision.utils.save_image(enhanced_image, result_path+image_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = '/data0/dsy/project/Enhancement_Detection_for_Aerial_Data/object_detection/datasets/dark_face/val/images/'
		os.environ['CUDA_VISIBLE_DEVICES']='5'
		file_list = os.listdir(filePath)
		print(file_list)
		#DarkLighter =  model_u_forme.enhancer().cuda()
		#DarkLighter.load_state_dict(torch.load('/data0/dsy/project/ULRE-Net/snapshots.pth'))
		DarkLighter =  model.enhancer().cuda()
		DarkLighter.load_state_dict(torch.load('/data0/dsy/project/ULRE-Net/snapshots_0/Epoch0.pth'))
		index = 0
		for file_name in file_list:
			image = glob.glob(filePath+file_name) 
			print(index)
			lowlight(image[0], DarkLighter)
			index = index + 1

