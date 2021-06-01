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
				  zeno_eps=0.1,
				  zeno_trim=0.4,
				  zeno_kloss=2,
				  zeno_alpha=0):
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
		zeno_trim (float): Fraction of clients to remove for trimmed Zeno
		zeno_kloss (int): Number of labels for top-k loss
		zeno_alpha (float): Multiplier for top-k loss term
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

		updates_device = next(iter(local_updates[0].values())).device
		dist = torch.zeros(len(local_updates), len(local_updates), device=updates_device)
		d = torch.zeros(len(local_updates), device=updates_device)

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
		val_loader = torch.utils.data.DataLoader(zeno_val_dataset, batch_size=zeno_batch_size, shuffle=True)
		images, labels = next(iter(val_loader))
		images, labels = images.to(device), labels.to(device)
		criterion = torch.nn.CrossEntropyLoss().to(device)

		global_model.zero_grad()
		global_outputs = global_model(images)
		global_loss = criterion(global_outputs, labels)
		global_loss.backward()

		for i, update in enumerate(local_updates):
			print(f'{i}------------------')
			# normalize g
			param_square = 0.0
			zeno_param_square = 0.0
			for k, zeno_param in global_model.named_parameters():
				param = update[k].detach()
				if zeno_param.requires_grad:
					param_square += param.square().sum()
					zeno_param_square += zeno_param.grad.detach().square().sum()
			c = -math.sqrt(zeno_param_square / param_square)
			for k, zeno_param in global_model.named_parameters():
				if zeno_param.requires_grad:
					param = update[k]
					param[:] = param * c
			print(f'param_square: {param_square:.5f}')
			print(f'zeno_param_square: {zeno_param_square:.5f}')
			# compute zeno score
			zeno_innerprod = 0.0
			zeno_square = param_square
			for k, zeno_param in global_model.named_parameters():
				param = update[k].detach()
				if zeno_param.requires_grad:
					zeno_innerprod += torch.sum(param * zeno_param.grad.detach())
			score = zeno_innerprod - zeno_rho * zeno_square + local_lr * zeno_eps
			if i >= 0:
				print(f'c: {c:.5f}')
				# print('innerprod:', zeno_innerprod.item())
				print(f'terms: {(zeno_innerprod).item():.5f} {(zeno_rho * zeno_square).item():.5f} {local_lr * zeno_eps:.5f}')
				print(f'score: {score.item():.5f}')
			if score >= 0.0:
				# rescale update
				for k, zeno_param in global_model.named_parameters():
					if zeno_param.requires_grad:
						param = update[k]
						param[:] = param / c
				updates.append(update)
				updates_sizes.append(local_sizes[i])

	elif defense_type == 'zeno':
		val_loader = torch.utils.data.DataLoader(zeno_val_dataset, batch_size=zeno_batch_size, shuffle=True)
		images, labels = next(iter(val_loader))
		images, labels = images.to(device), labels.to(device)
		criterion = torch.nn.CrossEntropyLoss().to(device)

		global_loss = batch_loss(global_model, images, labels, criterion)

		for i, update in enumerate(local_updates):
			# print(f'{i}------------------')
			local_model = create_local_model(global_model, update)
			param_square = grad_update_squared(global_model, update)
			local_loss = batch_loss(local_model, images, labels, criterion)

			# compute top k loss
			if zeno_alpha > 0.0:
				topk_loss = topk_loss_diff(global_model, local_model, zeno_kloss, images, labels, criterion)
			else:
				topk_loss = 0.0
			# compute zeno score
			score = global_loss - local_loss - zeno_rho * param_square + zeno_eps + topk_loss * zeno_alpha
			# print(f'terms: {global_loss.item():.5f} {local_loss.item():.5f} {(zeno_rho * param_square).item():.5f} {zeno_eps:.5f} {(topk_loss) * zeno_alpha:.5f} ')
			print(f'score {i}: {score.item():.5f}')
			if score >= 0.0:
				updates.append(update)
				updates_sizes.append(local_sizes[i])

	elif defense_type == 'zeno_trimmed':
		val_loader = torch.utils.data.DataLoader(zeno_val_dataset, batch_size=zeno_batch_size, shuffle=True)
		images, labels = next(iter(val_loader))
		images, labels = images.to(device), labels.to(device)
		criterion = torch.nn.CrossEntropyLoss().to(device)

		global_loss = batch_loss(global_model, images, labels, criterion)

		scores = []

		for i, update in enumerate(local_updates):
			# print(f'{i}------------------')
			local_model = create_local_model(global_model, update)
			param_square = grad_update_squared(global_model, update)
			local_loss = batch_loss(local_model, images, labels, criterion)

			score = global_loss - local_loss - zeno_rho * param_square
			# print(f'terms: {global_loss.item():.5f} {local_loss.item():.5f} {(zeno_rho * param_square).item():.5f} {zeno_eps:.5f}')
			print(f'score {i}: {score.item():.5f}')
			scores.append(score)
		
		num_chosen_clients = int(len(local_updates) * (1 - zeno_trim))
		update_indices = np.argsort(-np.array(scores))[:num_chosen_clients]
		updates = [local_updates[i] for i in update_indices]
		updates_sizes = [local_sizes[i] for i in local_sizes]
		print(update_indices)
	else:
		raise ValueError("Please specify a valid attack_type from ['fall' ,'little', 'gaussian'].")

	return updates, updates_sizes


def create_local_model(global_model, update):
	local_model = copy.deepcopy(global_model)
	w = local_model.state_dict()
	for key in w.keys():
		w[key] += update[key].type(w[key].dtype)
	return local_model

def batch_loss(model, images, labels, criterion):
	model.eval()
	with torch.no_grad():
		global_outputs = model(images)
		global_loss = criterion(global_outputs, labels)
	return global_loss

def grad_update_squared(global_model, update):
	param_square = 0.0
	for k, zeno_param in global_model.named_parameters():
		param = update[k].detach()
		if zeno_param.requires_grad:
			param_square += param.square().sum()
	return param_square

def per_label_loss(model, images, labels, criterion):
	n_classes = torch.unique(labels).max().item()
	losses = torch.zeros(n_classes)
	counts = torch.zeros(n_classes, dtype=int)
	with torch.no_grad():
		for c in range(n_classes):
			mask = c == labels
			counts[c] = mask.sum().item()
			if counts[c] > 0:
				losses[c] = criterion(model(images[mask]), labels[mask]) / counts[c]
			else:
				losses[c] = float('inf')
	return losses

def topk_loss(model, zeno_kloss, images, labels, criterion):
	losses = per_label_loss(model, images, labels, criterion)
	losses, indices = losses.sort()
	topk_loss = losses[:zeno_kloss].sum()
	# print(f'top labels: {indices[:zeno_kloss]} losses: {losses}')
	return topk_loss

def topk_loss_diff(global_model, local_model, zeno_kloss, images, labels, criterion):
	global_losses = per_label_loss(global_model, images, labels, criterion)
	local_losses = per_label_loss(local_model, images, labels, criterion)
	losses = global_losses - local_losses
	losses, indices = losses.sort(descending=True)
	topk_loss = losses[:zeno_kloss].sum()
	# print(f'top labels: {indices[:zeno_kloss]} losses: {losses}')
	return topk_loss