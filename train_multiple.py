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

np.random.seed(0)
torch.manual_seed(0)

class Trainer:
	'''
	Trains the model according to the given parameters
	'''
	def __init__(self):
		pass

	def plot_history(self, history, model_type, dataset_type, save_path):
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
			plt.xlabel("Number of epochs")
			plt.ylabel("Loss & Accuracy")
			plt.title("Performance Curves of %s on %s dataset"%(model_type, dataset_type))
			plt.savefig(os.path.join(save_path, "train_history.png"), dpi = 100, bbox_inches='tight')
			plt.clf()
			x = plt.imread(os.path.join(save_path, "train_history.png"))
			plt.axis('off')
			plt.imshow(x)
		except Exception as e:
			print(e)

	def setup_model(self, model_type, dataset_type, n_classes, pretrained_model = False, lr = None, flr = None):
		model_name, save_folder = create_path(model_type, dataset_type, self.version, not pretrained_model)
		model = selectModel(model_type, n_classes = n_classes)
		if pretrained_model:
			lr = flr
			load_model(model, pretrained_model)
# 		model.to(Config.device)
		optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.5, 0.999))
		LossCriterion = nn.CrossEntropyLoss()
		if self.debug: print(model_type, "setup ready")
		return model_name, save_folder, model, optimizer, LossCriterion
	
	def epoch_reset(self):
		for model_type in self.models_info.keys():
			if self.models_info[model_type]["stop"]: continue
			self.models_info[model_type]["training_accuracy_list"] = list()
			self.models_info[model_type]["validation_accuracy_list"] = list()
			self.models_info[model_type]["training_loss"] = 0.0
			self.models_info[model_type]["validation_loss"] = 0.0
			
			try:
				del self.models_info[model_type]["train_predictions"]
				del self.models_info[model_type]["loss"]
				del self.models_info[model_type]["val_predictions"]
				del self.models_info[model_type]["val_loss"]
			except:
				pass
	
	def epoch_status(self, epoch, batch_st, batch_end):
		print("Epoch: %03d / %03d | time/epoch: %d seconds"%(epoch, self.total_epochs, (batch_end - batch_st)))
		print("-"*60)
		for model_type in self.models_info.keys():
			if self.models_info[model_type]["stop"]: continue
			print("Model:", model_type)
			if epoch > 2:
				d = {
				"Training" : {"Loss -2": self.models_info[model_type]["history"]["train_loss"][-3],
							"Loss -1": self.models_info[model_type]["history"]["train_loss"][-2],
							"Loss" : self.models_info[model_type]["history"]["train_loss"][-1],
							"Accuracy -2": self.models_info[model_type]["history"]["train_accuracy"][-3],
							"Accuracy -1": self.models_info[model_type]["history"]["train_accuracy"][-2],
							"Accuracy" : self.models_info[model_type]["history"]["train_accuracy"][-1]
							 },
				"Validation" : {"Loss -2": self.models_info[model_type]["history"]["val_loss"][-3],
								"Loss -1": self.models_info[model_type]["history"]["val_loss"][-2],
								"Loss" : self.models_info[model_type]["history"]["val_loss"][-1],
								"Accuracy -2": self.models_info[model_type]["history"]["val_accuracy"][-3],
								"Accuracy -1": self.models_info[model_type]["history"]["val_accuracy"][-2],
								"Accuracy" : self.models_info[model_type]["history"]["val_accuracy"][-1]
							   },
			}
			else:
				d = {
				"Training" : {"Loss" : self.models_info[model_type]["history"]["train_loss"][-1],
							  "Accuracy" : self.models_info[model_type]["history"]["train_accuracy"][-1]
							 },
				"Validation" : {"Loss" : self.models_info[model_type]["history"]["val_loss"][-1],
								"Accuracy" : self.models_info[model_type]["history"]["val_accuracy"][-1]
							   },
				}
			print(pd.DataFrame(d).T)
			print("-"*40)
			
	def set_models(self, isTrain = True):
		for model_type in self.models_info.keys():
			if self.models_info[model_type]["stop"]: continue
			if isTrain: self.models_info[model_type]["model"].train()
			else: self.models_info[model_type]["model"].eval()
	
	def optimizers_reset(self):
		for model_type in self.models_info.keys():
			if self.models_info[model_type]["stop"]: continue
			self.models_info[model_type]["optimizer"].zero_grad()
			
	def save_best_models(self):
		for model_type in self.models_info.keys():
			if self.models_info[model_type]["stop"]: continue
			# Best validation loss
			if self.models_info[model_type]["best_val_loss"] >= self.models_info[model_type]["history"]["val_loss"][-1]:
				self.models_info[model_type]["best_val_loss"] = self.models_info[model_type]["history"]["val_loss"][-1]
				save_model(self.models_info[model_type]["model"], os.path.join(self.models_info[model_type]["save_path"], self.models_info[model_type]["model_name"].split(".")[0] + "_BEST_VL.pth"))
				print("%s: saved best model: VL"%(model_type))
			
			# Save model with highest validation accuracy
			if self.models_info[model_type]["best_val_accuracy"] <= self.models_info[model_type]["history"]["val_accuracy"][-1]:
				self.models_info[model_type]["best_val_accuracy"] = self.models_info[model_type]["history"]["val_accuracy"][-1]
				save_model(self.models_info[model_type]["model"], os.path.join(self.models_info[model_type]["save_path"], self.models_info[model_type]["model_name"].split(".")[0] + "_BEST_VA.pth"))
				print("%s: saved best model: VA"%(model_type))

	def check_early_stopping(self):
		for model_type in self.models_info.keys():
			if self.models_info[model_type]["stop"]: continue
			if self.models_info[model_type]["history"]["val_loss"][-1] > self.models_info[model_type]["history"]["val_loss"][-2]:
				self.models_info[model_type]["stop_penalty"]+=1
			else: self.models_info[model_type]["stop_penalty"] = 0

			if self.models_info[model_type]["stop_penalty"] > self.early_stopping_wait or self.models_info[model_type]["history"]["train_accuracy"][-1] >= self.max_accuracy:
				print("Waited for %d epochs"%(self.models_info[model_type]["stop_penalty"]))
				print("Last train accuracy %f", self.models_info[model_type]["history"]["train_accuracy"][-1])
				print("No improvement in validation accuracy. Stopping %s early"%(model_type))
				self.models_info[model_type]["stop"] = True

	def save_final(self, dataset_type):
		for model_type in self.models_info.keys():
			save_model(self.models_info[model_type]["model"], os.path.join(self.models_info[model_type]["save_path"], self.models_info[model_type]["model_name"].split(".")[0] + "_FINAL.pth"))
			pkl.dump(self.models_info[model_type]["history"], open(os.path.join(self.models_info[model_type]["save_path"], self.models_info[model_type]["model_name"].split(".")[0]+"_history.pkl"), "wb"))
			self.plot_history(self.models_info[model_type]["history"], model_type, dataset_type, self.models_info[model_type]["save_path"])
			
	def lr_decay(self, epoch):
		for model_type in self.models_info.keys():
			if self.models_info[model_type]["stop"]: continue
			if self.models_info[model_type]["decayLrPer"] and epoch % self.models_info[model_type]["decayLrPer"] == 0: adjust_learning_rate(self.models_info[model_type]["optimizer"], decayBy = self.models_info[model_type]["decayBy"])
		
	def train_models(self, model_list, dataset_type,
			pretrained_models = False,
			lr_list = 1e-3,
			flr_list = 1e-5,
			epochs = 300,
			batch_size = 32,
			starting_epoch = 1,
			version = 1,
			early_stopping_wait = 5,
			max_accuracy = 0.95,
			decayLrPer_list = 25,
			decayBy_list = 0.75,
			debug = True):
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
		self.version = version
		self.debug = debug
		self.total_epochs = epochs + starting_epoch - 1
		self.early_stopping_wait = early_stopping_wait
		self.max_accuracy = max_accuracy
		
		if isinstance(pretrained_models, bool) and not pretrained_models: pretrained_models = [False]*len(model_list)
		if not isinstance(lr_list, list): lr_list = [lr_list]*len(model_list)
		if not isinstance(flr_list, list): flr_list = [flr_list]*len(model_list)
		if not isinstance(decayLrPer_list, list): decayLrPer_list = [decayLrPer_list]*len(model_list)
		if not isinstance(decayBy_list, list): decayBy_list = [decayBy_list]*len(model_list)
		for item in [pretrained_models, lr_list, model_list, flr_list, decayLrPer_list, decayBy_list]:
			assert len(item) == len(model_list), "Parameters mismatch"
		
		# Prepare dataset
		ds = selectData(dataset_type, batch_size = batch_size)
		train_dataset, train_dataloader, val_dataset, val_dataloader = ds.getTrainData()
		n_classes = len(ds.classes)
		if debug: print("Data Loaded")
			
		# setup models
		self.models_info = dict()
		for model_type, pretrained_model, lr, flr, decayLrPer, decayBy in zip(model_list, pretrained_models, lr_list, flr_list, decayLrPer_list, decayBy_list):
			model_name, save_folder, model, optimizer, LossCriterion = self.setup_model(model_type, dataset_type, n_classes, pretrained_model, lr, flr)
			self.models_info[model_type] = dict()
			self.models_info[model_type]["stop"] = False
			self.models_info[model_type]["model_name"] = model_name
			self.models_info[model_type]["save_path"] = save_folder
			self.models_info[model_type]["model"] = model
			self.models_info[model_type]["optimizer"] = optimizer
			self.models_info[model_type]["loss_criterion"] = LossCriterion
			self.models_info[model_type]["decayBy"] = decayBy
			self.models_info[model_type]["decayLrPer"] = decayLrPer
			
			self.models_info[model_type]["history"] = {"train_loss" : list(), "val_loss": list(), "train_accuracy": list(), "val_accuracy": list()}
			self.models_info[model_type]["best_val_loss"] = 100
			self.models_info[model_type]["best_val_accuracy"] = 0.80
			self.models_info[model_type]["stop_penalty"] = 0
		
		# Start training
		for epoch in range(starting_epoch, self.total_epochs):
			batch_st = time()
			self.epoch_reset()
			
			# Train
			for batch_idx, batch_train_data in enumerate(train_dataloader, 0):
				self.set_models(isTrain = True)
				
				# move data to device
				train_images, train_labels = batch_train_data
				train_images = train_images.to(Config.device)
				train_labels = train_labels.to(Config.device)
				
				# clear gradients of optimizer
				self.optimizers_reset()
				
				# forward pass: compute predictions
				for model_type in self.models_info.keys():
					if self.models_info[model_type]["stop"]: continue
					self.models_info[model_type]["model"].to(Config.device)
					# predict
					self.models_info[model_type]["train_predictions"] = self.models_info[model_type]["model"](train_images)
				
					# calculate loss
					self.models_info[model_type]["loss"] = self.models_info[model_type]["loss_criterion"](self.models_info[model_type]["train_predictions"], train_labels)
					# backpropagate
					self.models_info[model_type]["loss"].backward()
					self.models_info[model_type]["optimizer"].step()
					
					# Record loss and accuracy
					self.models_info[model_type]["training_loss"] += (self.models_info[model_type]["loss"].item() * train_images.size(0))
					self.models_info[model_type]["training_accuracy_list"].append(calc_accuracy(train_labels, self.models_info[model_type]["train_predictions"]).item())
					self.models_info[model_type]["model"].to('cpu')
				
			# validate
			for val_batch_idx, batch_val_data in enumerate(val_dataloader):
				with torch.no_grad():
					self.set_models(isTrain = False)
					val_images, val_labels = batch_val_data
					val_images = val_images.to(Config.device)
					val_labels = val_labels.to(Config.device)
					
					for model_type in self.models_info.keys():
						if self.models_info[model_type]["stop"]: continue
						self.models_info[model_type]["model"].to(Config.device)
						self.models_info[model_type]["val_predictions"] = self.models_info[model_type]["model"](val_images)
						
						self.models_info[model_type]["val_loss"] = self.models_info[model_type]["loss_criterion"](self.models_info[model_type]["val_predictions"], val_labels)
						self.models_info[model_type]["validation_loss"] += (self.models_info[model_type]["val_loss"].item() * val_images.size(0))
						self.models_info[model_type]["validation_accuracy_list"].append(calc_accuracy(val_labels, self.models_info[model_type]["val_predictions"]).item())
						self.models_info[model_type]["model"].to('cpu')
			
			for model_type in self.models_info.keys():
				# Record to history
				self.models_info[model_type]["history"]["train_loss"].append(self.models_info[model_type]["training_loss"] / len(train_dataloader.sampler))
				self.models_info[model_type]["history"]["val_loss"].append(self.models_info[model_type]["validation_loss"] / len(val_dataloader.sampler))
				self.models_info[model_type]["history"]["train_accuracy"].append(np.mean(self.models_info[model_type]["training_accuracy_list"]))
				self.models_info[model_type]["history"]["val_accuracy"].append(np.mean(self.models_info[model_type]["validation_accuracy_list"]))
			
			# Print Status of the Epoch
			self.epoch_status(epoch, batch_st, time())
			# Save best models
			self.save_best_models()
			# Early stopping
			if epoch > 10: self.check_early_stopping()
			print("="*60)
			
			# Learning Rate Decay
			if np.sum(decayLrPer_list) > 1: self.lr_decay(epoch)
		
		# Save final model
		self.save_final(dataset_type)
		
		model_paths = [value["save_path"] for value in self.models_info.values()]
		
		try:
			del self.models_info, ds
			torch.cuda.empty_cache()
			gc.collect()
		except Exception as e:
			print("Could not clear the memory. Kill the process manually.")
			print(e)
			
		return model_paths