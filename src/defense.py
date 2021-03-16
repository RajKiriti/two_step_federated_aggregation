import torch
import numpy as np
import copy

from collections import OrderedDict

def defend_updates(local_updates, local_sizes, defense_type, trim_ratio=0.1, multi_krum=2):
	"""
	Robust Aggregation on top of the local updates.

	Args:
		local_updates (list) : Updates from all workers
		local_sizes (list) : Data sizes on all workers
		defense_type (str) : Robust Aggregation method to be used
		trim_ratio (float) : Trim ratio for trimmed mean aggregation
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
			sorted_stack, _ = torch.sort(stacked, dim=0)
			sorted_stack = sorted_stack[remove: len(local_updates) - remove] # Trimming the top and bottom updates after sorting
			temp_updates.append((k, [torch.flatten(i, start_dim=0, end_dim=1) for i in list(torch.split(sorted_stack, split_size_or_sections=1, dim=0))]))

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

	else:
		raise ValueError("Please specify a valid attack_type from ['fall' ,'little', 'gaussian'].")

	return updates, updates_sizes