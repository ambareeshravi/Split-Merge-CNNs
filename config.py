import torch
import os
import torchvision.transforms as transforms
import torch.optim as optim
from custom_transforms import *

class Config:
	# General Configuration
	Cuda = torch.cuda.is_available()
	ngpu = 0
	if Cuda: ngpu = 1
	device = torch.device("cuda:0" if Cuda else "cpu")
	print("-- Running on", device, "--")

	# Model configuration
	ImageSize = 224
	channels = 3
	LearningRate = 1e-3
	FineTuneLearningRate = 1e-5

	Epochs = 200
	BatchSize = 32
	starting_epoch = 1
	Betas = (0.5, 0.999)

	# Data configuration
	train_data_transform = transforms.Compose([        
						   transforms.RandomGrayscale(p = 0.25),
						   transforms.RandomHorizontalFlip(p=0.25),
						   transforms.RandomRotation(10),
						   transforms.Resize((ImageSize, ImageSize)),
						   transforms.ToTensor(),
						   transforms.RandomErasing(p=0.25, scale=(0.01, 0.01), ratio=(0.1, 1.0)),
						   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
							])
	
	test_data_transform = transforms.Compose([
						   transforms.Resize((ImageSize, ImageSize)),
						   transforms.ToTensor(),
						   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
								])