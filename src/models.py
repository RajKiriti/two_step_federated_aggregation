import torch
import torch.nn.functional as F
from torch import nn

class LR(nn.Module):

	"""
	Logistic Regression Model. (Convex objective)
	"""

	def __init__(self, dim_in, dim_out, seed):
		"""
		Args:
			dim_in (int) : Input dimension
			dim_out (int) : Output dimension
			seed (int) : Random seed value
		"""

		super(LR, self).__init__()

		torch.manual_seed(seed)

		self.dim_in = dim_in
		self.linear = nn.Linear(dim_in, dim_out)

	def forward(self, x):

		x = x.view(-1, self.dim_in) # Flattening the input
		x = self.linear(x)

		return F.log_softmax(x, dim=1)
	
class MLP(nn.Module):

	"""
	Multi Layer Perceptron with a single hidden layer.
	"""
	
	def __init__(self, dim_in, dim_hidden, dim_out, seed):
		"""
		Args:
			dim_in (int) : Input dimension
			dim_hidden (int) : # units in the hidden layer
			dim_out (int) : Output dimension
			seed (int) : Random seed value
		"""
		
		super(MLP, self).__init__()
		
		torch.manual_seed(seed)
		
		self.input = nn.Linear(dim_in, dim_hidden)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout()
		self.layer_hidden = nn.Linear(dim_hidden, dim_hidden)
		self.output = nn.Linear(dim_hidden, dim_out)

	def forward(self, x):
		
		x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
		x = self.input(x)
		x = self.relu(x)
		x = self.layer_hidden(x)
		x = self.relu(x)
		x = self.output(x)

		return F.log_softmax(x, dim=1)
	
class CNNMnist(nn.Module):

	"""
	2-layer CNN as used in (http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf).
		
	Note: TF code doesn't use dropout. 
	(https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py#L82)
	"""
	
	def __init__(self, seed):
		"""
		Args:
			seed (int) : Random seed value
		"""
		
		super(CNNMnist, self).__init__()
		
		torch.manual_seed(seed)
		
		self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
		self.fc1 = nn.Linear(7*7*64, 512)
		self.fc2 = nn.Linear(512, 10)

	def forward(self, x):
		
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2)
		x = x.view(-1, 7*7*64)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		
		return F.log_softmax(x, dim=1)


class CNNCIFAR(nn.Module):

	"""
	2-layer CNN as used in (http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf).
		
	Note: TF code doesn't use dropout. 
	(https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py#L82)
	"""
	
	def __init__(self, seed):
		"""
		Args:
			seed (int) : Random seed value
		"""
		
		super(CNNCIFAR, self).__init__()
		
		torch.manual_seed(seed)
		
		self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
		self.fc1 = nn.Linear(8*8*64, 512)
		self.fc2 = nn.Linear(512, 10)

	def forward(self, x):
		
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2)
		x = x.view(-1, 8*8*64)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		
		return F.log_softmax(x, dim=1)