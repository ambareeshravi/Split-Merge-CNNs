import os, cv2, shutil
import pickle as pkl, json
import numpy as np, pandas as pd

from time import time
from tqdm import tqdm
from glob import glob

import torch
from torch import nn

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import torchvision.datasets as dset
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, average_precision_score, confusion_matrix
import ml_metrics
import matplotlib.pyplot as plt
import gc

from custom_transforms import *
from config import Config
from models import *
from data import *

class Tester:
	'''
	Tests the model according to the given parameters
	'''
	def __init__(self):
		pass

	def get_metrics(self, y_act, y_pred):
	    print("ACCURACY: ", accuracy_score(y_act, y_pred))
	    print("ROC AUC Score: ",roc_auc_score(y_act, y_pred))
	    print("AVERAGE PRECISION SCORE: ", average_precision_score(y_act, y_pred))
	    print("MEAN AVERAGE PRECISON: ", " ".join(["| "+str(k)+"- "+str(ml_metrics.mapk(y_act.reshape(-1,1).tolist(), y_pred.reshape(-1,1).tolist(), k))[:5] for k in [5, 10, 50]]))
	    print("CLASSIFICATION REPORT:\n", classification_report(y_act, y_pred))
	    print("CONFUSION MATRIX:\n", confusion_matrix(y_act, y_pred))
	    print("-"*40)

	def test(self, model_path, debug = False):
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
		# Prepare dataset
		ds = selectData(model_path)
		test_dataset, test_dataloader = ds.getTestData()
		if debug: print("Data Loaded")

		# Prepare model
		test_model = selectModel(model_path)
		load_model(test_model, model_path)
		test_model.to(Config.device)
		test_model.eval()

		if debug: print("Model Ready")

		# Batch Testing
	    test_images = torch.Tensor()
	    test_labels = torch.Tensor()
	    for idx, data in enumerate(test_dataloader):
	        test_images = torch.cat((test_images, data[0]))
	        test_labels = torch.cat((test_labels, data[1].type(torch.FloatTensor)))
	        		   
	    print("Testing on %d samples"%(test_images.shape[0]))
	    # Predict
	    with torch.no_grad():
	        p = torch.Tensor()
	        for i in range(0, len(test_images), 10):
	            p = torch.cat((p, test_model(test_images[i:i+10].to(Config.device)).cpu()))
		
		# Reform predictions
	    y_act = np.squeeze(test_labels.detach().cpu().numpy())
	    predictions = p.detach().cpu().numpy()
	    if predictions.shape[-1] > 1:
	        y_pred = np.argmax(predictions, axis = -1)
	    else:
	        y_pred = np.round(np.squeeze(predictions))
	    
	    # Get classification metrics
	    self.get_metrics(y_act, y_pred)
	    del test_model, test_images, test_labels, p
	    gc.collect()

if __name__ == "__main__":
	tr = Tester()
	tr.test(model_type = "SplitMerge_XRay_V1_FINAL.pth", debug = True)