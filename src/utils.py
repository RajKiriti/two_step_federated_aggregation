import copy
import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def global_aggregate(global_optimizer, global_weights, local_updates, local_sizes, 
					global_lr=1., beta1=0.9, beta2=0.999, v=None, m=None, eps=1e-4, step=None):
	"""
	Aggregates the local client updates to find a focused global update.

	Args:
		global_optimizer (str) : Optimizer to be used for the steps
		global_weights (OrderedDict) : Initial state of the global model (which needs to be updated here)
		local_updates (list of OrderedDict) : Contains the update differences (delta) between global and local models
		local_sizes (list) : Sizes of local datasets for proper normalization
		global_lr (float) : Stepsize for global step
		beta1 (float) : Role of ``beta`` in FedAvgM, otheriwse analogous to beta_1 and beta_2 famous in literature for Adaptive methods
		beta2 (float) : Same as above
		v (OrderedDict) : Role of ``momentum`` in FedAvgM, else Adaptive methods
		m (OrderedDict) : Common in ADAM and YOGI.
		step (int) : Current epoch number to configure ADAM and YOGI properly
	"""
	
	total_size = sum(local_sizes)

	################################ FedAvg | SCAFFOLD ################################
	# Good illustration provided in SCAFFOLD paper - Equations (1). (https://arxiv.org/pdf/1910.06378.pdf)
	if global_optimizer in ['fedavg', 'scaffold']:
		
		w = copy.deepcopy(global_weights)

		for key in w.keys():
			for i in range(len(local_updates)):
				if global_optimizer == 'scaffold':
					w[key] += torch.mul(torch.div(local_updates[i][key], len(local_sizes)), global_lr).type(w[key].dtype)
				else:
					w[key] += torch.mul(torch.mul(local_updates[i][key], local_sizes[i]/total_size), global_lr).type(w[key].dtype)

		return w, v, m
	
	################################ FedAvgM ################################
	# Implementation similar to in (https://arxiv.org/pdf/1909.06335.pdf).
	elif global_optimizer == 'fedavgm':
		
		w = copy.deepcopy(global_weights)
		temp_v = copy.deepcopy(v)
		
		for key in w.keys():
			temp_v[key] = beta1*temp_v[key]
			for i in range(len(local_updates)):
				temp_v[key] -= torch.mul(torch.mul(local_updates[i][key], local_sizes[i]/total_size), global_lr)
			w[key] -= temp_v[key]
			
		return w, temp_v, m

	################################ FedAdam ################################
	# Adam from here : https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization.pdf
	elif global_optimizer in ['fedadam', 'fedyogi']:

		w = copy.deepcopy(global_weights)
		temp_v = copy.deepcopy(v)
		temp_m = copy.deepcopy(m)
		effective_lr = global_lr*math.sqrt(1 - beta2**step)/(1 - beta1**step)

		averaged_w = OrderedDict()
		for key in w.keys():
			averaged_w[key] = torch.zeros(w[key].shape, dtype=w[key].dtype)
			for i in range(len(local_updates)):
				averaged_w[key] += torch.mul(local_updates[i][key], local_sizes[i]/total_size)

		for key in w.keys():
			temp_m[key] = beta1*temp_m[key] + (1. - beta1)*averaged_w[key]

			if global_optimizer == 'fedadam':
				temp_v[key] = temp_v[key] - (1 - beta2)*(temp_v[key] - torch.pow(averaged_w[key], 2))
			else: #FedYogi
				temp_v[key] = temp_v[key] - (1 - beta2)*torch.mul(torch.sign(temp_v[key] - torch.pow(averaged_w[key], 2)), 
																			torch.pow(averaged_w[key], 2))

			w[key] += torch.mul(effective_lr, torch.div(temp_m[key], torch.add(eps, torch.pow(temp_v[key], 0.5))))

		return w, temp_v, temp_m

	else:

		raise ValueError('Check the global optimizer for a valid value.')
	
def network_parameters(model):
	"""
	Calculates the number of parameters in the model.

	Args:
		model : PyTorch model used after intial weight initialization
	"""
	total_params = 0
	
	for param in list(model.parameters()):
		curr_params = 1
		for p in list(param.size()):
			curr_params *= p
		total_params += curr_params
		
	return total_params

class DatasetSplit(Dataset):
	"""
	An abstract dataset class wrapped around Pytorch Dataset class.
	"""

	def __init__(self, dataset, idxs):

		self.dataset = dataset
		self.idxs = [int(i) for i in idxs]

	def __len__(self):

		return len(self.idxs)

	def __getitem__(self, item):
		
		image, label = self.dataset[self.idxs[item]]

		return torch.tensor(image), torch.tensor(label)

def test_inference(global_model, test_dataset, device, test_batch_size=128):
	"""
	Evaluates the performance of the global model on hold-out dataset.

	Args:
		global_model (model state) : Global model for evaluation
		test_dataset (tensor) : Hold-out data available at the server
		device (str) : One from ['cpu', 'cuda'].
		test_batch_size (int) : Batch size of the testing samples
	"""

	test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
	# criterion = nn.NLLLoss().to(device)
	criterion = nn.CrossEntropyLoss().to(device)
	global_model.eval()

	loss, total, correct = 0.0, 0.0, 0.0

	for batch_idx, (images, labels) in enumerate(test_loader):

		images, labels = images.to(device), labels.to(device)

		outputs = global_model(images)
		batch_loss = criterion(outputs, labels)
		loss += batch_loss.item()

		# Prediction
		_, pred_labels = torch.max(outputs, 1)
		pred_labels = pred_labels.view(-1)
		correct += torch.sum(torch.eq(pred_labels, labels)).item()
		total += len(labels)
	
	return correct/total, loss/total

def balanced_train_test_split(indices, labels, test_size, seed):
	"""
	Splits a dataset into train/test where the test set has the same number of samples with each label

	Args:
		indices (list): List of indices
		labels (list): List of labels corresponding to the indices
		test_size (float): fraction of dataset to partition into the test set
		seed (int): seed for random sampling
	"""
	n = len(indices)
	indices, labels = np.array(indices), np.array(labels)
	rng = np.random.default_rng(seed)

	shuffle = rng.permutation(n)
	indices = indices[shuffle]
	labels = labels[shuffle]

	unique_labels = np.unique(labels)
	num_each = int(n * test_size / len(unique_labels))
	counts = {i: 0 for i in unique_labels}

	pick = []
	for i in range(n):
		if counts[labels[i]] < num_each:
			pick.append(i)
			counts[labels[i]] += 1
	pick = np.array(pick)
	if sum([1 for c in counts.values() if c < num_each]):
		print('Unbalanced test set; test_size is too large')
	# print(np.unique(labels[pick], return_counts=True))
	train_set = np.delete(indices, pick)
	test_set = indices[pick]
	return train_set, test_set
