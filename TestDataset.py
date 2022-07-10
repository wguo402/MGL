########################################
# Created by andrew
# 2018/6/12
# predict label with a trained model 
########################################

import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
import os

class TestDataset(data.Dataset):
	def __init__(self, data_dir, transform):
		self.transform = transform
		self.data_dir = data_dir
		self.imgs = self.get_img_name(data_dir)

	def get_img_name(self, data_dir):
		img_names = os.listdir(self.data_dir)
		return img_names

	def get_img(self, img_path):
		img = Image.open(img_path).convert('RGB')
		return self.transform(img)

	def __getitem__(self, index):
		file = self.imgs[index]
		img_name = self.data_dir + file
		img = self.get_img(img_name)
		return img, file

	def __len__(self):
		return len(self.imgs)



		


