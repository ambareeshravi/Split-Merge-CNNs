import torch
from glob import glob

import os, cv2, shutil
import pickle as pkl, json
import numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader

import torchvision.datasets as dset
import torchvision.transforms as transforms

from config import Config

class XRayDataset:
	def __init__(self, data_path, total_samples = 3000):
		self.data_path = data_path

		# THIS IS FOR BINARY CLASSIFICATION. Change this if data is multiclass
		n1, n2 = tuple([len(glob(d + "/*")) for d in glob(os.path.join(data_path, "train/*"))])
		numDataPoints = n1+n2
		target = np.hstack((np.zeros(n1, dtype=np.int32), np.ones(n2, dtype=np.int32)))

		class_sample_count = np.array( [len(np.where(target == t)[0]) for t in np.unique(target)])
		weight = 1. / class_sample_count
		samples_weight = np.array([weight[t] for t in target])

		samples_weight = torch.from_numpy(samples_weight)
		samples_weight = samples_weight.double()
		self.sampler = torch.utils.data.WeightedRandomSampler(samples_weight, total_samples)

	def getTrainData(self):
		train_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "train"), transform = Config.train_data_transform)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.BatchSize, sampler=self.sampler)

		val_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "val"), transform = Config.test_data_transform)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=Config.BatchSize, shuffle=True)

		return train_dataset, train_dataloader, val_dataset, val_dataloader

	def getTestData(self):
		test_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "test"), transform = Config.test_data_transform)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=Config.BatchSize, shuffle=True)

		return  test_dataset, test_dataloader

class CTScanDataset:
	def __init__(self, data_path):
		self.data_path = data_path

	def getTrainData(self):
		train_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "train"), transform = Config.train_data_transform)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.BatchSize, sampler=self.sampler)

		val_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "val"), transform = Config.test_data_transform)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=Config.BatchSize, shuffle=True)

		return train_dataset, train_dataloader, val_dataset, val_dataloader

	def getTestData(self):
		test_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "test"), transform = Config.test_data_transform)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=Config.BatchSize, shuffle=True)

		return  test_dataset, test_dataloader