import torch
import numpy as np
import copy
import math

from collections import OrderedDict

def defend_updates(global_model,
				  local_updates,
				  local_sizes,
				  device,
				  local_lr,
				  defense_type,
				  trim_ratio=0.1,
				  multi_krum=2,
				  zeno_val_dataset=None,
				  zeno_batch_size=128,
				  zeno_rho=0.0001,
				  zeno_eps=0.1):
	"""
	Robust Aggregation on top of the local updates. This function destroys the original local_updates to save memory.

	Args:
		global_model (model state) : Copy of the global model
		local_updates (list) : Updates from all workers
		local_sizes (list) : Data sizes on all workers
		device (str) : One from ['cpu', 'cuda']
		local_lr (float) : Step-size for client training
		defense_type (str) : Robust Aggregation method to be used
		trim_ratio (float) : Trim ratio for trimmed mean aggregation
		multi_krum (float) : Number of clients to pick after krumming
		zeno_val_dataset (torch.utils.data.Dataset) : Dataset for computing Zeno++ score
		zeno_batch_size (int): Batch size for calculating gradient for Zeno++ score
		zeno_rho (float): Rho for Zeno++ score
		zeno_eps (float): Epsilon for Zeno++ score
	"""
	if len(local_updates) <= 0:
		raise ValueError('No local updates to aggregate.')

	updates = []
	updates_sizes = []

	if defense_type == 'median':

		w = OrderedDict()
		for k in local_updates[0].keys():
			w[k] = torch.zeros(local_updates[0][k].shape, dtype=local_updates[0][k].dtype)

		for k in w.keys():
			stacked = torch.stack([i[k] for i in local_updates]) # Stacking along 0th dimension
			median_val, _ = torch.median(stacked, dim=0)
			w[k] = median_val

		updates.append(w) # Will always have one local update
		updates_sizes.append(local_sizes[0]) # Won't matter as such

	elif defense_type == 'trimmed_mean': 
		
		remove = int(trim_ratio * len(local_updates))

		temp_updates = []
		for k in local_updates[0].keys():
			stacked = torch.stack([i[k] for i in local_updates]) # Stacking along 0th dimension
			for i in local_updates:
				i[k] = None
			sorted_stack, _ = torch.sort(stacked, dim=0)
			sorted_stack = sorted_stack[remove: len(local_updates) - remove] # Trimming the top and bottom updates after sorting
			temp_updates.append((k, [torch.flatten(i.detach().clone(), start_dim=0, end_dim=1) if len(i.shape) > 1 else i[0] for i in list(torch.split(sorted_stack, split_size_or_sections=1, dim=0))]))
			sorted_stack = None

		# (Input - 2 * remove) updates will be stored
		for i in range(len(local_updates) - 2*remove):
			curr = OrderedDict()
			for j in temp_updates:
				curr[j[0]] = j[1][i]
			updates.append(curr)
			updates_sizes.append(local_sizes[i]) # ----------------- Grey area ----------------- 

	elif defense_type == 'krum':
		
		remove = int(trim_ratio * len(local_updates))

		dist = torch.zeros(len(local_updates), len(local_updates))
		d = torch.zeros(len(local_updates))

		for i in range(len(local_updates)):
			for j in range(len(local_updates)):
				if i > j:
					dist[i][j] = dist[j][i]
				elif i == j:
					dist[i][j] = 0.0
				else:
					for k in local_updates[0].keys():
						dist[i][j] += torch.norm(local_updates[i][k] - local_updates[j][k])
			dist[i], _ = torch.sort(dist[i]) # Sorting the squared 2-norm distances for `i` worker
			d[i] = torch.sum(dist[i][1 : len(local_updates) - remove - 1])
		d, idxs = torch.sort(d) # Sorting the clients with lowest d[i]

		for i in range(multi_krum):
			updates.append(local_updates[idxs[i]])
			updates_sizes.append(local_sizes[idxs[i]])

	elif defense_type == 'zeno++':
		val_loader = torch.utils.data.DataLoader(zeno_val_dataset, batch_size=zeno_batch_size, shuffle=False)
		images, labels = next(iter(val_loader))
		images, labels = images.to(device), labels.to(device)
		criterion = torch.nn.CrossEntropyLoss().to(device)

		global_outputs = global_model(images)
		global_loss = criterion(global_outputs, labels)
		global_loss.backward()

		for i, update in enumerate(local_updates):
			# normalize g
			param_square = 0.0
			zeno_param_square = 0.0
			for k, zeno_param in global_model.named_parameters():
				param = update[k].detach()
				if zeno_param.requires_grad:
					param_square += param.square().sum()
					zeno_param_square += zeno_param.grad.detach().square().sum()
			c = -math.sqrt(zeno_param_square / param_square)
			print('\nc:', c)
			for k, zeno_param in global_model.named_parameters():
				if zeno_param.requires_grad:
					param = update[k]
					param[:] = param * c

			# compute zeno score
			zeno_innerprod = 0.0
			zeno_square = param_square
			for k, zeno_param in global_model.named_parameters():
				param = update[k].detach()
				if zeno_param.requires_grad:
					zeno_innerprod += torch.sum(param * zeno_param.grad.detach())
			print('innerprod:', zeno_innerprod)
			print('terms:', local_lr * zeno_innerprod, zeno_rho * zeno_square, local_lr * zeno_eps)
			score = local_lr * zeno_innerprod - zeno_rho * zeno_square + local_lr * zeno_eps
			print('score:', score)
			if score >= 0.0:
				# rescale update
				for k, zeno_param in global_model.named_parameters():
					if zeno_param.requires_grad:
						param = update[k]
						param[:] = -local_lr * param
				updates.append(update)
				updates_sizes.append(local_sizes[i])

	else:
		raise ValueError("Please specify a valid attack_type from ['fall' ,'little', 'gaussian'].")

	return updates, updates_sizes