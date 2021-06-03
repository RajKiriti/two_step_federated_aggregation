# Secure Byzantine-Robust Distributed Learning via Clustering

This repository is the official implementation of Secure Byzantine-Robust Distributed Learning via Clustering. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To run experiments on Shakespeare, download the [LEAF framework](https://github.com/TalwalkarLab/leaf)
and run ```./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8 --smplseed 0 --spltseed 0``` inside the directory ```leaf/data/shakespeare```.
Then copy the ```leaf/data/shakespeare``` folder into the ```data``` folder of this repository.

## Experiments

To run the experiements in the paper, first edit ```config.json``` to specify hyperparameters.
Hyperparameters not specified in the paper are set to their default values.
Then, run this command:

```train
python main.py
```

Results will be outputed to a csv file in the ```results``` directory.

## Compute used

We ran all of our experiments on a single core of a NVIDA Tesla K80 GPU.


## Code referenced

We referenced implementations of existing models in ```src/models``` and included license information for that code.
