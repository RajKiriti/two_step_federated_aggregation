import numpy as np
from collections import Counter

def iid(dataset, num_users, seed):
	"""
	Divides the given dataset in a IID fashion into specified number of users.

	Args:
		dataset (tensor) : dataset to be partitioned
		num_users (int) : # users to be created
		seed (int) : Random seed value
	"""
	np.random.seed(seed)
	
	num_items = int(len(dataset) / num_users)
	rem_items = len(dataset) % num_users
	if rem_items == 0:
		print("Each user will get %d samples from the training set."%(num_items))
	else:
		print("Each user will get %d samples from the training set. %d samples are discarded."%(num_items, rem_items))

	user_groups = {} 
	all_idxs = list(range(len(dataset)))
	
	for i in range(num_users):
		user_groups[i] = list(np.random.choice(all_idxs, num_items, replace=False))
		all_idxs = list(set(all_idxs) - set(user_groups[i]))
	
	return user_groups

def non_iid(dataset, num_users, num_shards_user, seed):
	"""
	Divides the given dataset in a non-IID fashion, after dividing into shards.

	Args:
		dataset (tensor) : dataset to be partitioned
		num_users (int) : # users to be created
		num_shards_user (int) : # shards to be given to each user after pathological generation
		seed (int) : Random seed value
	"""
	np.random.seed(seed)

	num_shards = num_users * num_shards_user
	assert len(dataset) % num_shards == 0, "Error with number of shards as (number of images per shard) * \
											(number of shards) doesn't equal (number of images)"
	num_images = int(len(dataset) / num_shards)
	print("Each shard contains %d images."%(num_images))
	
	idx_shard = list(range(num_shards))
	user_groups = {i: np.array([], dtype='int64') for i in range(num_users)}
	idxs = np.arange(num_shards * num_images)
	labels = dataset.targets.numpy()
	
	# Sort labels and corresponding indexes
	idxs_labels = np.vstack((idxs, labels))
	idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
	idxs = idxs_labels[0,:]
	
	for i in range(num_users):
		random_shards = set(np.random.choice(idx_shard, num_shards_user, replace=False))
		idx_shard = list(set(idx_shard) - random_shards)
		for r in random_shards:
			user_groups[i] = np.concatenate((user_groups[i], idxs[r*num_images : (r+1)*num_images]), axis=0)

	return user_groups

"""

def non_iid_dirichlet(dataset, num_users, priors, alpha, num_classes, seed):

	# Divides the given dataset in a non-IID fashion, based on test data distribution.
	# Similar to the sampling technique in https://arxiv.org/pdf/1909.06335.pdf.

	np.random.seed(seed)

	num_items = int(len(dataset) / num_users)
	rem_items = len(dataset) % num_users
	if rem_items == 0:
		print("Each user will get %d samples from the training set."%(num_items))
	else:
		print("Each user will get %d samples from the training set. %d samples are discarded."%(num_items, rem_items))

	priors = list(np.array(priors)*alpha)
	dist = np.random.dirichlet(alpha=priors, size=num_users)
	labels = dataset.targets
	
	print((dist.sum(axis=0)*num_items).astype(int))

	labels_classes = [[idx for idx, j in enumerate(list(labels)) if j==i] for i in range(num_classes)]
	
	user_groups = {}
	assert len(dist) == num_users
	for idx, d in enumerate(dist):
		num_images = [int(round(j*num_items)) for j in d]
		assert len(num_images) == num_classes
		user_sampled = []
		for i in range(num_classes):
			if num_images[i] > len(labels_classes[i]):
				curr_sampled = np.random.choice(labels_classes[i], len(labels_classes[i]), replace=False)
				labels_classes[i] = list(set(labels_classes[i]) - set(curr_sampled))
				num_images[i] = num_images[i] - len(labels_classes[i])
				user_sampled = user_sampled + list(curr_sampled)
				# Pick remaining from the class with highest remaining samples
				while num_images[i] > 0:
					highest_rem = np.argmax(np.array([len(i) for i in labels_classes]))
					if num_images[i] > len(labels_classes[highest_rem]):
						curr_sampled = np.random.choice(labels_classes[highest_rem], len(labels_classes[highest_rem]), replace=False)
						num_images[i] = num_images[i] - len(labels_classes[highest_rem])
					else:
						curr_sampled = np.random.choice(labels_classes[highest_rem], num_images[i], replace=False)
						num_images[i] = 0
						break
					labels_classes[highest_rem] = list(set(labels_classes[highest_rem]) - set(curr_sampled))
					user_sampled = user_sampled + list(curr_sampled)
#				print(len(labels_classes[highest_rem]), num_images[i] - len(labels_classes[i]))
			else:
				curr_sampled = np.random.choice(labels_classes[i], num_images[i], replace=False)
				labels_classes[i] = list(set(labels_classes[i]) - set(curr_sampled))
				user_sampled = user_sampled + list(curr_sampled)
		user_groups[idx] = user_sampled
		
	return user_groups

"""