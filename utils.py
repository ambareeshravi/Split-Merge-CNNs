import torch
from torch import nn
from sklearn.metrics import accuracy_score
from models import *
from data import *
from glob import glob

def save_model(model, model_name):
	'''
	Saves model object to a path
	'''
	if ".tar" not in model_name: model_name += ".tar"
	torch.save({'state_dict': model.state_dict()}, model_name)

def load_model(model, model_name):
	'''
	Load model parameters to the model object
	'''
	if ".tar" not in model_name: model_name += ".tar"
	checkpoint = torch.load(model_name, map_location = 'cpu')
	model.load_state_dict(checkpoint['state_dict'])

def adjust_learning_rate(optimizer, decayBy = 0.9):
	for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr'] * decayBy

def weights_init(m, initializer_type = "kaiming_uniform"):
	'''
	Initializes the weigts of the network
	'''
	initializer_type = initializer_type.lower()
	if "kaiming" in initializer_type:
		if "uniform" in initializer_type:
			init_fn = nn.init.kaiming_uniform_
		elif "normal" in initializer_type:
			init_fn = nn.init.kaiming_normal_
	if "xavier"	in initializer_type:
		if "uniform" in initializer_type:
			init_fn = nn.init.xavier_uniform_
		elif "normal" in initializer_type:
			init_fn = nn.init.xavier_normal_
	if isinstance(m, nn.Conv2d):
		init_fn(m.weight.data)
# 		init_fn(m.bias.data)
# 	if isinstance(m, nn.BatchNorm2d):
# 		init_fn(m.weight.data, 1.0, 0.02)
# 		init_fn(m.bias.data, 0)

def calc_accuracy(labels, predictions, argmaxReq = False):
	'''
	Calculates the accuracy given labels and predictions
	'''
	labels = labels.detach().cpu().numpy()
	predictions = predictions.detach().cpu().numpy()
	if argmaxReq: predictions = np.argmax(predictions, axis = -1)
	return accuracy_score(labels.flatten(), np.round(predictions.flatten()))

def selectModel(model_type, n_classes = 1):
	'''
	selects the model type and returns an object

	Args:
		model_type - model name as <str>
	Returns:
		model object
	Exception:
		-
	'''
	model_type = model_type.lower()
	if "mobilenet" in model_type: return get_mobilenet(n_classes = n_classes)
	elif "inception" in model_type: return get_inception(n_classes = n_classes)
	elif "normal" in model_type: return NormalCounterpart(n_classes = n_classes)
	elif "split" in model_type: return SplitMerge(n_classes = n_classes)
	else: raise TypeError("[ERROR]: Incorrect Model Type")

def selectData(dataset_type):
	'''
	selects the dataset type and creates an object

	Args:
		dataset_type - dataset name as <str>
	Returns:
		dataset object
	Exception:
		-
	'''
	dataset_type = dataset_type.lower()
	if "xray" in dataset_type: return XRayDataset()
	elif "ctscan" in dataset_type: return CTScanDataset()
	else: raise TypeError("[ERROR]: Incorrect Dataset Type")
		
def getNParams(model):
	params_dict = dict()
	params_dict["Total"] = sum(p.numel() for p in model.parameters())
	params_dict["Trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
	params_dict["Non-Trainable"] = params_dict["Total"] - params_dict["Trainable"]
	return params_dict

def get_save_folder(model_type, data_type, version):
	model_version = "_".join([model_type, data_type, "V%d"%(version)])
	model_name = "_".join([model_type, data_type, "V%d.pth"%(version)])
	save_folder = os.path.join("models/", model_version)
	return model_name, save_folder

def create_path(model_type, data_type, version, createNew = True):
	# Path Configuration
	model_name, save_folder = get_save_folder(model_type, data_type, version)
	try: os.mkdir(save_folder)
	except:
		if createNew:
			version = max([int(d[-1]) for d in glob(save_folder[:-2] + "*")]) + 1
			model_name, save_folder = get_save_folder(model_type, data_type, version)
			os.mkdir(save_folder)
		else:
			pass
	return model_name, save_folder