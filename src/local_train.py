import copy
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.optimizers import StochasticControl
from src.utils import DatasetSplit

class LocalUpdate(object):
	
	def __init__(self, dataset, idxs, device, train_test_split=0.8,
				train_batch_size=32, test_batch_size=32, attack=None,
				num_classes=None, flip_eps=None):
		"""
		Args:
			dataset (tensor) : Global data
			idxs (list) : List of indexes corresponding to the global data for making it local
			device (str) : One from ['cpu', 'cuda'].
			train_test_split (float) : Proportion of client data to be split for training and testing
			train_batch_size (int) : Batch size of the training samples
			test_batch_size (int) : Batch size of the testing samples
			attack (str) : Type of attack, one of ['label_flip']
			num_classes (int) : Total number of classes for label_flip attack
		"""
	
		self.device = device
		self.train_test_split = train_test_split
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.attack = attack
		self.num_classes = num_classes
		self.flip_eps = flip_eps
		# self.criterion = nn.NLLLoss().to(self.device) # Default criterion set to NLL loss function
		self.criterion = nn.CrossEntropyLoss().to(self.device)

		self.train_test(dataset, list(idxs)) # Creating train and test splits

	def train_test(self, dataset, idxs):
		"""
		Creates the train and test loader using the provided dataset and indexes pertaining to local client

		Args:
			dataset (tensor) : Global data
			idxs (list) : List of indexes corresponding to the global data for making it local
		"""

		self.train_loader = DataLoader(DatasetSplit(dataset, idxs[:int(self.train_test_split * len(idxs))]), 
												batch_size=self.train_batch_size, shuffle=True)
		self.test_loader = DataLoader(DatasetSplit(dataset, idxs[int(self.train_test_split * len(idxs)):]), 
												batch_size=self.test_batch_size, shuffle=False)

	def local_opt(self, optimizer, lr, epochs, global_model, momentum=0.5, mu=0.01, client_controls=[], 
				server_controls=[], global_round=0, client_no=0, batch_print_frequency=100, lambda_val=None):
		"""
		Local client optimization in the form of updates/steps.

		Args:
			optimizer (str) : Local optimizer to be used for training
			lr (float) : step-size for client training
			epochs (int) : # local steps to be taken
			global_model (model state) : Initial global model
			momentum (float) : Momentum parameter for SGD
			mu (float) : Coefficient for the proximal term in FedProx
			client_controls (list) : Control variates for the client
			server_controls (list) : Control variates for the server
			global_round (int) : Current global federated round number
			client_no (int) : Current client number
			batch_print_frequency (int) : Epoch cut-off for printing batch-level measures
			lambda_val (OrderedDict) : Lambda values for the sampled client
		"""
		
		global_params = [global_model.state_dict()[key] for key in global_model.state_dict()]
		server_controls_list = [server_controls[key] for key in server_controls.keys()]
		client_controls_list = [client_controls[key] for key in client_controls.keys()]
		lambda_val = [lambda_val[key] for key in lambda_val.keys()]
		
		# Set model to ``train`` mode
		local_model = copy.deepcopy(global_model)
		local_model.train()

		# Set local optimizer
		if optimizer == 'sgd':
			opt = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
		elif optimizer == 'adam':
			opt = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)
		elif optimizer in ['fedprox', 'scaffold', 'fedalm']:
			opt = StochasticControl(local_model.parameters(), lr=lr, mu=mu, weight_decay=1e-4)
		else:
			raise ValueError("Please specify a valid value for the optimizer from ['sgd', 'adam', 'fedprox', 'scaffold'].")

		epoch_loss = []
							 
		for epoch in range(epochs):
			
			batch_loss = []
			
			for batch_idx, (images, labels) in enumerate(self.train_loader):
				
				# Label flip attack
				if self.attack == 'label_flip':
					labels = (self.num_classes - 1) - labels # Assuming labels are 0 to (snum_classes - 1)

				images, labels = images.to(self.device), labels.to(self.device)
				local_model.zero_grad()
				log_probs = local_model(images)
				loss = self.criterion(log_probs, labels)
				loss.backward()
				if optimizer in ['fedprox', 'scaffold', 'fedalm']:
					opt.step(optimizer, global_params, server_controls_list, client_controls_list, lambda_val)
				else:
					opt.step()

				if (batch_idx+1) % batch_print_frequency == 0:
					msg = '| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
					print(msg.format(global_round, epoch, batch_idx * len(images), len(self.train_loader.dataset),
						100. * batch_idx / len(self.train_loader), loss.item()))

				batch_loss.append(loss.item())

			epoch_loss.append(sum(batch_loss)/len(batch_loss))

		# Finding the local update in parameters
		local_changes = OrderedDict()
		for k in global_model.state_dict():
			if self.attack == 'sign_flip':
				local_changes[k] = self.flip_eps * (local_model.state_dict()[k] - global_model.state_dict()[k])
			else:
				local_changes[k] = local_model.state_dict()[k] - global_model.state_dict()[k]

		# Updating local client variates
		control_changes = OrderedDict()
		if optimizer == 'scaffold':
			for key in client_controls.keys():
				control_changes[key] = torch.mul(-1., client_controls[key])
				client_controls[key] = client_controls[key] - server_controls[key] - local_changes[key]# torch.div(local_params[key], (lr*local_epochs))
				control_changes[key] += client_controls[key]

		return local_changes, control_changes, client_controls, epoch_loss[-1], len(self.train_loader)

	def inference(self, global_model):
		"""
		Returns the inference accuracy and loss.

		Args:
			global_model (model state) : Global model for evaluation
		"""
		
		global_model.eval()
		loss, total, correct = 0.0, 0.0, 0.0

		for batch_idx, (images, labels) in enumerate(self.test_loader):

			images, labels = images.to(self.device), labels.to(self.device)

			outputs = global_model(images)
			batch_loss = self.criterion(outputs, labels)
			loss += batch_loss.item()

			_, pred_labels = torch.max(outputs, 1)
			pred_labels = pred_labels.view(-1)
			correct += torch.sum(torch.eq(pred_labels, labels)).item()
			total += len(labels)

		return correct/total, loss