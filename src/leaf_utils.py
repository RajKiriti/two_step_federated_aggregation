import json
import re
import numpy as np
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):

    def __init__(self, data_dir, num_clients, train):
        train_data_dir = os.path.join(data_dir, 'shakespeare', 'data', 'train')
        test_data_dir = os.path.join(data_dir, 'shakespeare', 'data', 'test')
        self.clients, self.groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
        if train:
            data = train_data
        else:
            data = test_data
        self.clients = self.clients[:num_clients]
        self.groups = self.groups[:num_clients]
        self.data = []
        self.targets = []
        self.user_ranges = []
        start_idx = 0
        for client in list(data.values())[:num_clients]:
            x, y = client['x'], client['y']
            self.user_ranges.append(list(range(start_idx, start_idx + len(x))))
            start_idx += len(x)
            self.data.extend(list(zip(x, y)))
            self.targets.extend([ALL_LETTERS.find(yi) for yi in y])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self._format_data(*self.data[idx])

    def _format_data(self, xi, yi):
        new_xi = torch.tensor([ALL_LETTERS.find(c) for c in xi])
        new_yi = ALL_LETTERS.find(yi)
        return (new_xi, new_yi)

    def user_groups(self):
        return self.user_ranges

# Code from LEAF utils 

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data

# ------------------------
# utils for shakespeare dataset

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string
    
    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices
