import numpy as np
import torch
from torch import nn
from torchvision import models

def get_inception(n_classes = 2):
	'''
	Creates an instance of Incpetion V3 model
	'''
	return nn.Sequential(
		models.inception_v3(pretrained=False, num_classes = n_classes, aux_logits = False),
		nn.Softmax()
	)

def get_mobilenet(n_classes=2):
	'''
	Creates an instance of MobilNet V2 model
	'''
	return nn.Sequential(
		models.MobileNetV2(num_classes=n_classes),
		nn.Softmax()
	)


class NormalCounterpart(nn.Module):
	'''
	Implements the Normal Counter part of the Split-Merge CNN classification architecture
	'''
	def __init__(self, n_classes = 1, ngpu = 1):
		'''
		Initializes the class parameters for the model

		Args:
			output_nodes - Number of classes as <int>
			For Binary Classiciation 1/2 units can be used
		Returns:
			-
		'''
		super(NormalCounterpart, self).__init__()

		self.filters_count = np.array([64, 64, 128, 256, 128, 64], dtype = np.uint16)
		self.reduction_count = np.array([64,32,16], dtype = np.uint16)
		
		self.ngpu = ngpu
		final_activation = nn.Sigmoid()
		if n_classes > 1: final_activation = nn.Softmax()
		
		# Block 1
		self.lconv1 = nn.Conv2d(3, self.filters_count[0], 3, 1)
		self.act_block1 = self.BN_R(self.filters_count[0])
		
		# Block 2
		self.lconv2 = nn.Conv2d(self.filters_count[0], self.filters_count[1], 3, 2)
		self.act_block2 = self.BN_R(self.filters_count[1])
		
		# Block 3
		self.lconv3 = nn.Conv2d(self.filters_count[1], self.filters_count[2], 4, 3)
		self.act_block3 = self.BN_R(self.filters_count[2])
		
		# Block 4
		self.lconv4 = nn.Conv2d(self.filters_count[2], self.filters_count[3], 4, 3)
		self.act_block4 = self.BN_R(self.filters_count[3])
		
		# Block 5
		self.lconv5 = nn.Conv2d(self.filters_count[3], self.filters_count[4], 3, 2)
		self.act_block5 = self.BN_R(self.filters_count[4])
		
		# Block 6
		self.lconv6 = nn.Conv2d(self.filters_count[4], self.reduction_count[0], 3, 2)
		self.act_block6 = self.BN_R(self.reduction_count[0])
		
		# Classification head
		self.classifier = nn.Sequential(
			nn.MaxPool2d(2),
			nn.Dropout(0.25),
			nn.Flatten(),
			nn.Dropout(0.25),
			nn.Linear(self.reduction_count[0], self.reduction_count[1]),
			nn.Dropout(0.25),
			nn.Linear(self.reduction_count[1], self.reduction_count[2]),
			nn.Dropout(0.25),
			nn.Linear(self.reduction_count[2], n_classes),
			nn.Sigmoid()
		)

	def BN_R(self, n, p = 0.2):
		'''
		Activation Block: ReLU(BN(x))

		Args:
			n - number of output channels as <int>
			p - probability of dropout as <float>
		Returns:
			<nn.Sequential> block
		'''
		return nn.Sequential(
			nn.BatchNorm2d(n, momentum = None, track_running_stats = False),
			nn.ReLU6(inplace = True), # Found ReLU6 to be performing better thatn ReLU (Improvement)
			nn.Dropout2d(p) # Slows down training but better generalization (Improvement)
		)
				
	def forward(self, x):
		'''
		Forward pass for the Normal Counterpart of the split merge model

		Args:
			x - inputs as <torch.Tensor>
		Returns:
			output as <torch.Tensor>
		'''
		l1_op = self.act_block1(self.lconv1(x))
		l2_op = self.act_block2(self.lconv2(l1_op))
		l3_op = self.act_block3(self.lconv3(l2_op))
		l4_op = self.act_block4(self.lconv4(l3_op))
		l5_op = self.act_block5(self.lconv5(l4_op))
		l6_op = self.act_block6(self.lconv6(l5_op))
		op = self.classifier(l6_op)
		
		return op

class SplitMerge(nn.Module):
	'''
	Implements Split-Merge CNN classification architecture
	'''
	def __init__(self, n_classes = 1, ngpu = 1):
		'''
		Initializes the class parameters for the model

		Args:
			output_nodes - Number of classes as <int>
			For Binary Classiciation 1/2 units can be used
		Returns:
			-
		'''
		super(SplitMerge, self).__init__()

		# Effect seen if GPU is available
		self.ngpu = ngpu

		final_activation = nn.Sigmoid()
		if n_classes > 1: final_activation = nn.Softmax()

		# Depends on the complexity of the data

		self.filters_count = np.array([64, 64, 128, 256, 128, 64], dtype = np.uint16)
		self.reduction_count = np.array([64,32,16], dtype = np.uint16)		
		
		# SM Block 1
		self.lconv1 = nn.Conv2d(3, self.filters_count[0], 3, 1)
		self.rconv1 = nn.Conv2d(3, self.filters_count[0], 5, 1, padding_mode='replicate', padding=1)
		self.mconv1 = nn.Conv2d(3, self.filters_count[0], 2, 1, dilation=2)
		
		self.act_block1 = self.BN_R(self.filters_count[0])
		
		# SM Block 2
		self.lconv2 = nn.Conv2d(self.filters_count[0], self.filters_count[1], 3, 2)
		self.rconv2 = nn.Conv2d(self.filters_count[0], self.filters_count[1], 4, 2)
		
		self.act_block2 = self.BN_R(self.filters_count[1])
		
		# SM Block 3
		self.lconv3 = nn.Conv2d(self.filters_count[1], self.filters_count[2], 4, 3)
		self.rconv3 = nn.Conv2d(self.filters_count[1], self.filters_count[2], 5, 3)
		
		self.act_block3 = self.BN_R(self.filters_count[2])
		
		# SM Block 4
		self.lconv4 = nn.Conv2d(self.filters_count[2], self.filters_count[3], 4, 3)
		self.rconv4 = nn.Conv2d(self.filters_count[2], self.filters_count[3], 5, 3)
		
		self.act_block4 = self.BN_R(self.filters_count[3])
		
		# SM Block 5
		self.lconv5 = nn.Conv2d(self.filters_count[3], self.filters_count[4], 3, 2)
		self.rconv5 = nn.Conv2d(self.filters_count[3], self.filters_count[4], 4, 2, padding_mode='replicate', padding=1)
		
		self.act_block5 = self.BN_R(self.filters_count[4])
		
		# SM Block 6
		self.lconv6 = nn.Conv2d(self.filters_count[4], self.reduction_count[0], 3, 2)
		self.rconv6 = nn.Conv2d(self.filters_count[4], self.reduction_count[0], 4, 2, padding_mode='replicate', padding=1)
		
		self.act_block6 = self.BN_R(self.reduction_count[0])
		
		# Classification head
		self.classifier = nn.Sequential(
			# Feature size reduction
			nn.MaxPool2d(2),
			nn.Dropout(0.25),
			nn.Flatten(),
			nn.Dropout(0.25),

			# FC layers
			nn.Linear(self.reduction_count[0], self.reduction_count[1]),
			nn.Dropout(0.25),
			nn.Linear(self.reduction_count[1], self.reduction_count[2]),
			nn.Dropout(0.25),
			nn.Linear(self.reduction_count[2], n_classes),
			final_activation
		)

	def BN_R(self, n, p = 0.2):
		'''
		Activation Block: ReLU(BN(x))

		Args:
			n - number of output channels as <int>
			p - probability of dropout as <float>
		Returns:
			<nn.Sequential> block
		'''
		return nn.Sequential(
			nn.BatchNorm2d(n, momentum = None, track_running_stats = False),
			nn.ReLU6(inplace = True), # Found ReLU6 to be performing better thatn ReLU (Improvement)
			nn.Dropout2d(p) # Slows down training but better generalization (Improvement)
		)
				
	def forward(self, x):
		'''
		Forward pass for the split merge model

		Args:
			x - inputs as <torch.Tensor>
		Returns:
			output as <torch.Tensor>
		'''
		l1_a = self.act_block1((self.lconv1(x) + self.mconv1(x) + self.rconv1(x)))
		l2_a = self.act_block2((self.lconv2(l1_a) + self.rconv2(l1_a)))
		l3_a = self.act_block3((self.lconv3(l2_a) + self.rconv3(l2_a)))
		l4_a = self.act_block4((self.lconv4(l3_a) + self.rconv4(l3_a)))
		l5_a = self.act_block5((self.lconv5(l4_a) + self.rconv5(l4_a)))
		l6_a = self.act_block6((self.lconv6(l5_a) + self.rconv6(l5_a)))
		op = self.classifier(l6_a)
		
		return op