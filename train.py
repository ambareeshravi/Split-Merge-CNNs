import os, cv2, shutil
import pickle as pkl, json
import numpy as np
import pandas as pd

from time import time
from tqdm import tqdm
from glob import glob

import torch
from torch import nn
from PIL import Image

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import gc

from custom_transforms import *
from config import Config
from models import *
from data import *
from utils import *

class Trainer:
	'''
	Trains the model according to the given parameters
	'''
	def __init__(self):
		pass

	def plot_history(self, history, model_type, dataset_type):
		'''
		Plots graph with model history

		Args:
			history -  dict with parameter as key and list of values
			model_type - model name as <str>
			dataset_type - dataset name as <str>
		Returns:
			-
		Exception:
			general exception
		'''
		try:
			epochs = list(range(len(history["train_loss"])))
			plt.plot(epochs, history["train_loss"], label = "Train Loss")
			plt.plot(epochs, history["val_loss"], label = "Validation Loss")
			plt.plot(epochs, history["train_accuracy"], label = "Train Accuracy")
			plt.plot(epochs, history["val_accuracy"], label = "Validation Accuracy")
			plt.legend()
			plt.xlabel("Number of Epochs")
			plt.ylabel("Loss & Accuracy")
			plt.title("Performance Curves of %s on %s dataset"%(model_type, dataset_type))
			plt.savefig(os.path.join(self.save_folder, "train_history.png"), dpi = 100, bbox_inches='tight')
			plt.clf()
			x = plt.imread(os.path.join(self.save_folder, "train_history.png"))
			plt.axis('off')
			plt.imshow(x)
		except Exception as e:
			print(e)

	def train(self, model_type, dataset_type, pretrained_model = False, lr = 1e-4, flr = 1e-5, epochs = 300, batch_size = 32, starting_epoch = 1, version = 1, early_stopping_wait = 5, max_accuracy = 0.95, decayLrPer = 10, debug = True):
		'''
		Trains the models

		Args:
			model_type - model name as <str>
			dataset_type - dataset name as <str>
			pretrained_model - path to any pretrained model as <str>
		Returns:
			-
		Exception:
			-
		'''
		# Override config's default parameter values
		Config.LearningRate = lr
		Config.FineTuneLearningRate = flr
		Config.Epochs = epochs
		Config.BatchSize = batch_size
		Config.starting_epoch = starting_epoch
		
		self.model_name, self.save_folder = create_path(model_type, dataset_type, version, not pretrained_model)
		# Prepare dataset
		ds = selectData(dataset_type)
		train_dataset, train_dataloader, val_dataset, val_dataloader = ds.getTrainData()
		if debug: print("Data Loaded")

		# Prepare model
		n_classes = 1
		model = selectModel(model_type, n_classes = n_classes)
# 		model.apply(weights_init)
		if pretrained_model:
			Config.LearningRate = Config.FineTuneLearningRate
			load_model(model, pretrained_model)
			if debug: print("Loaded Pre-Trained weights")
		model.to(Config.device)
		if debug: print("Model Ready")

		# Define optimizer and Loss function
		optimizer = torch.optim.Adam(model.parameters(), lr = Config.LearningRate, betas=Config.Betas)
		LossCriterion = nn.BCELoss()
		if n_classes > 2:
			LossCriterion = nn.CrossEntropyLoss()
		
		# Create variables to record history and store the best models
		history = {"train_loss" : list(), "val_loss": list(), "train_accuracy": list(), "val_accuracy": list()}
		best_val_loss, best_val_accuracy = 100, 0.80
		stop_penalty = 0
		
		# Start training
		for epoch in range(Config.starting_epoch, Config.Epochs + Config.starting_epoch):
			batch_st = time()
			
			accuracy_list = list()
			val_accuracy_list = list()
			train_loss, validation_loss = 0.0, 0.0
			prev_val_loss, prev_val_accuracy = 100.0, 0.0
			
			# Iterate over batches
			for batch_idx, data in enumerate(train_dataloader, 0):
				model.train()
				images, labels = data
				if n_classes == 1: labels = labels.type(torch.FloatTensor)
				labels = labels.to(Config.device)
				# clear gradients of optimizer
				optimizer.zero_grad()
				
				# forward pass: compute predictions
				predictions = model(images.to(Config.device))
				if n_classes == 1: predictions = predictions.flatten()
				# calculate loss
				loss = LossCriterion(predictions, labels)
				# backpropagate
				loss.backward()
				optimizer.step()
				
				train_loss += (loss.item() * images.size(0))
				
				accuracy = calc_accuracy(labels, predictions, n_classes > 1)
				accuracy_list.append(accuracy.item())
				
			# validate
			for val_batch_idx, batch_val_data in enumerate(val_dataloader):
				with torch.no_grad():
					model.eval()
					val_images, val_labels = batch_val_data
					val_images = val_images.to(Config.device)
					if n_classes == 1: val_labels = val_labels.type(torch.FloatTensor)
					val_labels = val_labels.to(Config.device)
					val_predictions = model(val_images)
					if n_classes == 1: val_predictions = val_predictions.flatten()
					# Calculate Loss
					val_loss = LossCriterion(val_predictions, val_labels)
					validation_loss += (val_loss.item() * val_images.size(0))
					val_accuracy = calc_accuracy(val_labels, val_predictions, n_classes > 1)
					val_accuracy_list.append(val_accuracy.item())
			
			# Calculate average loss					
			train_loss = train_loss/len(train_dataloader.sampler)
			validation_loss = validation_loss/len(val_dataloader.sampler)
			
			# Record to history
			history["train_loss"].append(train_loss)
			history["val_loss"].append(validation_loss)
			history["train_accuracy"].append(np.mean(accuracy_list))
			history["val_accuracy"].append(np.mean(val_accuracy_list))
			
			# Print Status of the Epoch
			print("Epoch: %03d / %03d \nTraining: LOSS: %.4f | Accuracy: %.4f | time/epoch: %d seconds \nValidation: TOTAL: %0.4f | Accuracy: %0.4f" % (epoch, Config.Epochs + Config.starting_epoch - 1, train_loss, np.mean(accuracy_list), time() - batch_st, validation_loss, np.mean(val_accuracy_list)))

			# Save model with lowest validation loss
			if best_val_loss >= validation_loss:
				save_model(model, os.path.join(self.save_folder, self.model_name.split(".")[0] + "_BEST_VL.pth"))
				best_val_loss = validation_loss
				print("Saved best model: VL")
			
			# Save model with highest validation accuracy
			if best_val_accuracy <= np.mean(val_accuracy_list):
				best_val_accuracy = np.mean(val_accuracy_list)
				save_model(model, os.path.join(self.save_folder, self.model_name.split(".")[0] + "_BEST_VA.pth"))
				print("Saved best model: VA")
			
			print("-"*40)

			# Early stopping
			if val_loss > prev_val_loss: stop_penalty += 1
			else: stop_penalty = 0
				
			if stop_penalty > early_stopping_wait or np.mean(accuracy_list) >= max_accuracy:
				print("Waited for %d epochs"%(stop_penalty))
				print("Last train accuracy %f", np.mean(accuracy_list))
				print("No improvement in validation accuracy. Stopping early")
				break
				
			prev_val_loss = val_loss
			prev_val_accuracy = np.mean(val_accuracy_list)
			
			if decayLrPer and epoch % decayLrPer == 0: adjust_learning_rate(optimizer, decayBy = 0.9)
		
		# Save final model
		save_model(model, os.path.join(self.save_folder, self.model_name.split(".")[0] + "_FINAL.pth"))
		pkl.dump(history, open(os.path.join(self.save_folder, self.model_name.split(".")[0]+"_history.pkl"), "wb"))
		self.plot_history(history, model_type, dataset_type)
		try:
			del model, ds, optimizer, LossCriterion
			torch.cuda.empty_cache()
			gc.collect()
		except Exception as e:
			print("Could not clear the memory. Kill the process manually.")
			print(e)
    
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_type', required=True, default='NeoNet8', help='Type of the model to train')
	parser.add_argument('--dataset_type', required=True, default='XRay', help='Type of the dataset to train on')
	args = parser.parse_args()
	
	tr = Trainer()
	tr.train(model_type = args.model_type, dataset_type = args.dataset_type)
	models_path = tr.save_folder
	
	del tr
	gc.collect()
	
	from test_models import *
	# Test
	ts = Tester()
	for model_path in glob(os.path.join(models_path, "*.tar")):    
		try:
			print("Testing:", model_path)
			ts.test(model_path)
			print("-"*20)
		except:
			pass
	del ts
	gc.collect()