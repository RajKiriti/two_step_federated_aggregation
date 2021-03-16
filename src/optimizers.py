from torch.optim.optimizer import Optimizer, required
import copy
import torch

class StochasticControl(Optimizer):
	
	"""
	Perturbed Gradient Descent as per TF implementation of FedProx (https://arxiv.org/abs/1812.06127)
	When mu = 0, it is similar to SGD.

	SCAFFOLD as per the Algorithm in (https://arxiv.org/pdf/1910.06378.pdf).
	"""

	def __init__(self, params, lr=required, mu=0, weight_decay=0):
		"""
		params (iterable) : model.parameters() in PyTorch
		lr (float) : learning rate
		mu (float) : coefficient for the proximal term
		"""
		
		if lr is not required and lr < 0.0:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if weight_decay < 0.0:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

		# Setting the passed parameters to the default ``params``
		defaults = dict(lr=lr, mu=mu, weight_decay=weight_decay)
			
		#print(params, defaults)
		super(StochasticControl, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(StochasticControl, self).__setstate__(state)

	def step(self, opt=None, global_params=[], server_controls=[], 
			client_controls=[] , lambda_val=[], closure=None):
		"""
		Performs a single optimization step.
		
		Args:
			opt (str) : Local update optimizer to be used from ['fedprox', 'scaffold']
			global_params (list) : initial params
			c_server (list) : Control variates for the central server
			c_client (list) : Control variates for the client
			lambda_val (list) : Update parameter in ALM
			closure (callable, optional) : A closure that reevaluates the model and returns the loss.
		"""
		assert type(global_params) == list, "global_params must be a list."
		assert type(server_controls) == list, "c_server must be a list."
		assert type(client_controls) == list, "c_client must be a list."
		
		loss = None
		if closure is not None:
			loss = closure()
		
		for group in self.param_groups:
			weight_decay = group['weight_decay']
			mu = group['mu']

			for idx, p in enumerate(group['params']):
				if p.grad is None:
					continue
				d_p = p.grad
				if weight_decay != 0:
					d_p = d_p.add(p, alpha=weight_decay)

				if opt == 'fedprox':
					########################### FedProx Update ###########################
					p.data = p.data - group['lr']*(d_p + mu*(p.data - global_params[idx]))
					######################################################################

				elif opt == 'fedalm':
					########################### FedALM Update ##############################################
					p.data = p.data - group['lr']*(d_p + mu*(p.data - global_params[idx]) + lambda_val[idx])
					########################################################################################

				elif opt == 'scaffold':
					################################ Scaffold Update ################################
					p.data = p.data - group['lr']*(d_p + server_controls[idx] - client_controls[idx])
					#################################################################################
					
				else:
					raise ValueError('Please specify a valid value for the local optimizer.')

		return loss