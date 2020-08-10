import torch
from torch import nn
from sklearn.metrics import accuracy_score

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

def weights_init(m):
	'''
	Initializes the weigts of the network
	'''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def calc_accuracy(labels, predictions, argmaxReq = False):
	'''
	Calculates the accuracy given labels and predictions
	'''
    labels = labels.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    if argmaxReq: predictions = np.argmax(predictions, axis = -1)
    return accuracy_score(labels.flatten(), np.round(predictions.flatten()))