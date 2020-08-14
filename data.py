import torch
from glob import glob

import os, cv2, shutil
import pickle as pkl, json
import numpy as np, pandas as pd

import torchvision.datasets as dset
import torchvision.transforms as transforms

from config import Config

class XRayDataset:
	def __init__(self, data_path = "../../COVID19_Preemptive/data/Images/XRay_gan/", total_samples = 3000):
		self.data_path = data_path

	def getTrainData(self):
		train_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "train"), transform = Config.train_data_transform)
		
		class_sample_count = np.array([len(os.listdir(os.path.join(os.path.join(self.data_path, "train"), directory))) for directory in train_dataset.classes])
		target = list()
		for class_idx, count in enumerate(class_sample_count):
			target += [class_idx] * count
		target = np.array(target)

		weight = 1. / class_sample_count
		samples_weight = np.array([weight[t] for t in target])

		samples_weight = torch.from_numpy(samples_weight)
		samples_weight = samples_weight.double()
		total_samples = int(np.min(class_sample_count) * len(class_sample_count))
		sampler = torch.utils.data.WeightedRandomSampler(samples_weight, 2600)
		
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.BatchSize, sampler=sampler)

		val_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "val"), transform = Config.test_data_transform)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=Config.BatchSize, shuffle=True)

		return train_dataset, train_dataloader, val_dataset, val_dataloader

	def getTestData(self):
		test_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "test"), transform = Config.test_data_transform)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=Config.BatchSize, shuffle=True)

		return  test_dataset, test_dataloader

class CTScanDataset:
	def __init__(self, data_path = "../../COVID19_Preemptive/data/Images/CT_Scans/"):
		self.data_path = data_path

	def getTrainData(self):
		train_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "train"), transform = Config.train_data_transform)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.BatchSize, shuffle=True)

		val_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "val"), transform = Config.test_data_transform)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=Config.BatchSize, shuffle=True)

		return train_dataset, train_dataloader, val_dataset, val_dataloader

	def getTestData(self):
		test_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "test"), transform = Config.test_data_transform)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=Config.BatchSize, shuffle=True)

		return  test_dataset, test_dataloader
	
class DogBreeds:
	def __init__(self, data_path = "../../datasets/DogBreeds/"):
		self.data_path = data_path

	def getTrainData(self):
		train_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "train"), transform = Config.train_data_transform)
		
		class_sample_count = np.array([len(os.listdir(os.path.join(os.path.join(self.data_path, "train"), directory))) for directory in train_dataset.classes])
		target = list()
		for class_idx, count in enumerate(class_sample_count):
			target += [class_idx] * count
		target = np.array(target)

		weight = 1. / class_sample_count
		samples_weight = np.array([weight[t] for t in target])

		samples_weight = torch.from_numpy(samples_weight)
		samples_weight = samples_weight.double()
		sampler = torch.utils.data.WeightedRandomSampler(samples_weight, int(np.mean(class_sample_count) * len(class_sample_count)))
		
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.BatchSize, sampler=sampler)

		val_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "val"), transform = Config.test_data_transform)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=Config.BatchSize, shuffle=True)

		return train_dataset, train_dataloader, val_dataset, val_dataloader

	def getTestData(self):
		test_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "test"), transform = Config.test_data_transform)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=Config.BatchSize, shuffle=True)
		return  test_dataset, test_dataloader