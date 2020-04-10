import numpy
import torch
import torchvision
import os
import torch
import numpy as np
from tqdm import tqdm

from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class SklearnDataset(torch.utils.data.Dataset):
    """docstring for OlivettiDataset"""
    
    def __init__(self, fetch_function, *fct_args, **fct_kwargs):
        self.dataset = fetch_function(*fct_args, **fct_kwargs)
        self.targets = self.dataset.target
        self.images = self.dataset.images.reshape((len(self.dataset.images), -1))
        self.classes = numpy.unique(self.targets)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

    def __len__(self):
        return len(self.images)

class AnnDataDataset(torch.utils.data.Dataset):
    def __init__(self, adata, X_field=None, X_expressions_transform=None, include_obs=None, obs_transforms=None, pre_densify=False, oversample_obs_key=None):
        """
        Args:
            -X_expressions_transform tranformation to apply to gene expression
            -include_obs a list of obs to include
            -obs_transform a dict of transforms to apply to the obs
            -X_field a filed name in obsm, if left to None, adata.X will be used
        """
        # self.adata = adata
        self.X_expressions_transform = X_expressions_transform
        self.include_obs = include_obs
        self.obs_transforms = obs_transforms
        self.pre_densify = pre_densify
        self.X_field = X_field
        self.oversample_obs_key = oversample_obs_key
        self._preprocess(adata)

    def _preprocess(self, adata):
        if self.X_field is None:
            self.X_data = adata.X
        else :
            self.X_data = adata.obsm[self.X_field]
        
        if self.X_expressions_transform:
            self.X_data = self.X_expressions_transform(self.X_data)
        
        if self.pre_densify and not isinstance(self.X_data, numpy.ndarray):
            self.X_data = torch.tensor( self.X_data.todense(), dtype=torch.float )
        else:
            self.X_data = torch.tensor( self.X_data, dtype=torch.float )

        print("data shape:", self.X_data.shape)
        
        self.obs_data = None 
        if self.include_obs :
            self.obs_data = {}
            for obs_key in self.include_obs:
                self.obs_data[obs_key] = adata.obs[obs_key].values
                if self.obs_transforms is not None :
                    if obs_key in self.obs_transforms:
                        self.obs_data[obs_key] = self.obs_transforms[obs_key]( self.obs_data[obs_key] )
                self.obs_data[obs_key] = torch.tensor(self.obs_data[obs_key])
    
        if self.oversample_obs_key is not None:
            unique_values = adata.obs[self.oversample_obs_key].unique()
            self.oversample_indexes = {
                "keys": [],
                "indexes":[],
            }
            for u_key in unique_values:
                self.oversample_indexes["keys"].append(u_key)
                self.oversample_indexes["indexes"].append(numpy.where(adata.obs[self.oversample_obs_key] == u_key)[0])
            
    def __len__(self):
        return self.X_data.shape[0]

    def __getitem__(self, idx):
        if self.oversample_obs_key is not None:
            obs_type = numpy.random.randint(0, len(self.oversample_indexes["keys"]))
            idx = numpy.random.choice(self.oversample_indexes["indexes"][obs_type])
            
        X_data = self.X_data[idx]

        if not self.pre_densify and not isinstance(self.X_data, numpy.ndarray):
            X_data = torch.tensor(X_data.todense(), dtype=torch.float)[0]
        
        sample = { 'X_data': X_data }
        
        for obs_key in self.include_obs:
            sample[obs_key] = self.obs_data[obs_key][idx]

        return sample

def is_images(name):
    return name.lower() in ["mnist", "olivetti"]

def get_label_encoder(data):
    from sklearn import preprocessing
    
    le = preprocessing.LabelEncoder()
    y = le.fit(data)

    return le

def bernoulli_corrupt(dropout_rate):
    def _do(data):
        # print("coo", dropout_rate)
        if dropout_rate > 0:

            idx = data > 0
            pixels = data[idx]
            mask = numpy.random.binomial(1, dropout_rate, len(pixels))
            mask = torch.tensor(mask, dtype=pixels.dtype)  
            data[idx] = pixels * mask
        return data
    return _do

def float32(data):
    return data.type(torch.float32)

def max_normalize(data):
    res = data / torch.max(data)
    return res

def raster_reshape(data):
    in_size = data[0].shape[0] * data[0].shape[1]
    return data.view((-1, in_size))[0]

def load_pytorch_image_dataset(name, batch_size, bernoulli_dropout_rate=0):
    from sklearn import preprocessing

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        float32,
        max_normalize,
        bernoulli_corrupt(bernoulli_dropout_rate),
        raster_reshape
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        float32,
        max_normalize,
        raster_reshape
    ])
    
    dataset_fct = getattr(torchvision.datasets, name)
    train_dataset = dataset_fct(root='./dataset_%s/' % name, train=True, transform=train_transforms, download=True)
    test_dataset  = dataset_fct(root='./dataset_%s/' % name, train=False, transform=test_transforms, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    nb_class = len(train_dataset.classes)
    le = preprocessing.LabelEncoder()
    le.fit(range(nb_class))

    in_size = train_dataset.data[0].shape[0] * train_dataset.data[0].shape[1]

    return {
        "name": name,
        "datasets": {
            "train": train_dataset,
            "test": test_dataset,
        },
        "loaders": {
            "train": train_loader,
            "test": test_loader,
        },
        "shapes": {
            "nb_class": nb_class,
            "input_size": in_size,
            "total_size": len(train_dataset) + len(test_dataset),
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
        },
        "batch_formater": lambda x: (x[0], x[1]),
        "label_encoding": le,
        "sample_scale": (0, 1)
    }

def load_mnist(batch_size, bernoulli_dropout_rate=0):
    return load_pytorch_image_dataset("MNIST", batch_size, bernoulli_dropout_rate)

def load_olivetti(batch_size):
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn import preprocessing

    train_dataset = SklearnDataset(fetch_olivetti_faces)
    # train_dataset.images = train_dataset.images - train_dataset.images.mean(axis=0)
    # train_dataset.images -= train_dataset.images.mean(axis=1).reshape(train_dataset.images.shape[0], -1)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    nb_class = len(train_dataset.classes)
    le = preprocessing.LabelEncoder()
    le.fit(range(nb_class))
    in_size = train_dataset.images[0].shape[0]

    return {
        "name": "olivetti",
        "datasets": {
            "train": train_dataset,
            "test": None,
        },
        "loaders": {
            "train": train_loader,
            "test": None,
        },
        "shapes": {
            "nb_class": nb_class,
            "input_size": in_size,
            "total_size": len(train_dataset),
            "train_size": len(train_dataset),
            "test_size": None,
        },
        "batch_formater": lambda x: (x[0], x[1]),
        "label_encoding": le,
        "sample_scale": (0, 1)
    }


def make_single_cell_dataset(batch_size, condition_field, adata, dataset_name, pre_densify=True, oversample=True, X_field=None,):
    
    le = get_label_encoder(adata.obs[condition_field])
    print(condition_field, adata.obs[condition_field].unique())
    nb_class = len(le.classes_)

    if oversample:
        oversample_obs_key = condition_field
    else:
        oversample_obs_key = None

    train_dataset = AnnDataDataset(
        adata,
        include_obs=[condition_field],
        obs_transforms = {
            condition_field: le.transform
        },
        pre_densify=pre_densify,
        oversample_obs_key=oversample_obs_key,
        X_field=X_field
    )
    
    in_size = train_dataset.X_data.shape[1]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    scale = ( torch.max(train_dataset.X_data), torch.min(train_dataset.X_data))
    print("range of sample inputs:", scale)

    return {
        "name": dataset_name,
        "adata": adata,
        "datasets": {
            "train": train_dataset,
            "test": None,
        },
        "loaders": {
            "train": train_loader,
            "test": None,
        },
        "shapes": {
            "nb_class": nb_class,
            "input_size": in_size,
            "total_size": len(train_dataset),
            "train_size": len(train_dataset),
            "test_size": None,
        },
        "batch_formater": lambda x: (x["X_data"], x[condition_field]),
        "label_encoding": le,
        "sample_scale": scale
    }

def load_single_cell(batch_size, condition_field, filepath, dataset_name, max_norm=True):
    from . import single_cell
    adata = single_cell.load_10x_dataset(filepath)
    return make_single_cell_dataset(batch_size, condition_field, adata, dataset_name)
