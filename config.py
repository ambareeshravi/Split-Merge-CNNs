import torch
import os
import torchvision.transforms as transforms
import torch.optim as optim

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
	LearningRate = 1e-4

	Epochs = 100
	BatchSize = 64
	starting_epoch = 1
	
	# Path Configuration
	version  = 1
	data_type = "X-Ray"
	model_type = "Split-Merge"
	model_version = "_".join([model_type, data_type, "V%d"%(version)])
	model_name = "_".join([model_type, data_type, "V%d.pth"%(version)])
	save_folder = os.path.join("models/", model_version)
	try: os.mkdir(save_folder)
	except: pass

	# Data configuration
	data_path = "data/" # Path to the dataset
	
	train_data_transform = transforms.Compose([        
						   transforms.RandomGrayscale(p = 0.25),
						   transforms.RandomHorizontalFlip(p=0.25),
						   transforms.RandomRotation(10),
						   transforms.Resize(ImageSize),
						   transforms.CenterCrop(ImageSize),
						   transforms.ToTensor(),
						   transforms.RandomErasing(p=0.25, scale=(0.01, 0.01), ratio=(0.1, 1.0)),
						   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
							])
	
	test_data_transform = transforms.Compose([
							   transforms.Resize(ImageSize),
							   transforms.CenterCrop(ImageSize),
							   transforms.ToTensor(),
							   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
								])