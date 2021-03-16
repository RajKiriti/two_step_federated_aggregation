import torch
import numpy as np
import copy

from collections import OrderedDict

def attack_updates(global_weights, defense_type, attack_type, 
					local_byz_updates, local_byz_sizes, little_std, fall_eps):
	"""
	Attacks the benign updates and converts to byzantine.

	Args:
		global_weights (OrderedDict) : State of the global model
		defense_type (str) : Assumed aggregation method
		attack_type (str) : Method of attacking the updates
		local_byz_updates (list) : Updates from byzantine workers
		local_byz_sizes (list) : Corresponding data lengths for byzantine workers
		little_std (float) : Standard deviation for `A Little Is Enough`
		fall_eps (float) : Epsilon to be used for the `Fall of Empires`
	"""
	m = 0.0
	s = 0.0
	w_byzantine = None

	if attack_type == 'fall':

		# if defense_type == 'krum':
		# 	epsilon = - 1.0

		# elif defense_type == 'trimmed_mean' or defense_type == 'median':
		# 	epsilon = - 5.0

		# else:
		# 	raise ValueError('Please specify a valid value for defense type from [`krum`, `trimmed_mean`, `median`].')

		w_byzantine = OrderedDict()
		for k in global_weights.keys():
			w_byzantine[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
		
		for k in w_byzantine.keys():
			for i in range(len(local_byz_updates)):
					w_byzantine[k] += torch.mul(local_byz_updates[i][k], float(local_byz_sizes[i]/sum(local_byz_sizes)))
			w_byzantine[k] = fall_eps * w_byzantine[k]

	elif attack_type in ['gaussian', 'little']:

		w_byzantine = OrderedDict()
		m = OrderedDict()
		s = OrderedDict()

		for k in global_weights.keys():
			w_byzantine[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
			m[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
			s[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)

		for k in global_weights.keys():
			for i in range(len(local_byz_updates)):
				a = torch.mul(local_byz_updates[i][k], local_byz_sizes[i]/sum(local_byz_sizes))
				m[k] += a
				s[k] += torch.mul(a, a)
			s[k]= (torch.abs(s[k] - torch.mul(m[k], m[k]))) ** 0.5

			if attack_type == 'little':
				w_byzantine[k] = m[k] - little_std * s[k]

	else:
		raise ValueError("Please specify a valid attack_type from ['fall' ,'little', 'gaussian'].")

	return w_byzantine, m, s