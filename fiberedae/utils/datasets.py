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

class BasicDataset(object):
    """docstring for BasicDataset"""
    def __init__(self, samples, targets):
        super(BasicDataset, self).__init__()
        self.samples = samples
        self.targets = targets

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

    def __len__(self):
        return len(self.samples)

        
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

        self.obs_data = {} 
        if self.include_obs :
            self.obs_data = {}
            if len(self.include_obs) > 0:
                for obs_key in self.include_obs:
                    self.obs_data[obs_key] = adata.obs[obs_key].values
                    if self.obs_transforms is not None :
                        if obs_key in self.obs_transforms:
                            self.obs_data[obs_key] = self.obs_transforms[obs_key]( self.obs_data[obs_key] )
                    self.obs_data[obs_key] = torch.tensor(self.obs_data[obs_key])
        else :
            self.obs_data["zeros"] = torch.zeros(self.X_data.shape[0]).long()

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
        
        if self.include_obs :
            for obs_key in self.include_obs:
                sample[obs_key] = self.obs_data[obs_key][idx]
        else:
            sample["zeros"] = self.obs_data["zeros"][idx]

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

def load_pytorch_image_dataset(name, batch_size, bernoulli_dropout_rate=0,num_workers=8):
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

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
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

def load_mnist(batch_size, bernoulli_dropout_rate=0,num_workers=8):
    return load_pytorch_image_dataset("MNIST", batch_size, bernoulli_dropout_rate)

def load_olivetti(batch_size):
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn import preprocessing

    train_dataset = SklearnDataset(fetch_olivetti_faces)
    # train_dataset.images = train_dataset.images - train_dataset.images.mean(axis=0)
    # train_dataset.images -= train_dataset.images.mean(axis=1).reshape(train_dataset.images.shape[0], -1)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
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


def make_single_cell_dataset(batch_size, condition_field, adata, dataset_name, pre_densify=True, oversample=True, X_field=None,num_workers=8):
    if condition_field:
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
        batch_formater = lambda x: (x["X_data"], x[condition_field])
    else :
        nb_class = 1
        oversample_obs_key = None
        train_dataset = AnnDataDataset(
            adata,
            pre_densify=pre_densify,
            X_field=X_field
        )
        batch_formater = lambda x: ( x["X_data"], x["zeros"] )
        le = lambda x: 0
    
    in_size = train_dataset.X_data.shape[1]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workerss)
    
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
        "batch_formater": batch_formater,
        "label_encoding": le,
        "sample_scale": scale
    }

def load_single_cell(batch_size, condition_field, filepath, dataset_name, backup_url=None):
    from . import single_cell
    adata = single_cell.load_10x_dataset(filepath, backup_url=backup_url)
    return make_single_cell_dataset(batch_size, condition_field, adata, dataset_name)

def load_blobs(n_samples, nb_class, nb_dim, batch_size, mask_class, dropout_rate=0, random_state=1234,num_workers=8):
    """Make a blobs (isotropic gaussians) datasets"""
    from sklearn.datasets import make_blobs
    blobs, targets = make_blobs(n_samples=n_samples, centers=nb_class, n_features=nb_dim, random_state=1234)
    blobs = blobs - np.min(blobs)
    blobs = blobs / np.max(blobs)

    blobs_clean = blobs
    if dropout_rate > 0:
        mask = np.random.binomial(1, 1-dropout_rate, blobs.shape)
        blobs = mask * blobs
    
    torch_blobs = torch.tensor( blobs, dtype=torch.float )
    torch_targets = torch.tensor( targets )
    if mask_class:
        torch_targets = torch.zeros_like(torch_targets)

    dataset = BasicDataset(torch_blobs, torch_targets)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return {
        "name": "Blobs",
        "datasets": {
            "train": {
                "dataset": dataset,
                "samples": blobs,
                "clean_samples": blobs_clean,
                "unmasked_targets": targets,
            },
            "test": None,
        },
        "loaders": {
            "train": train_loader,
            "test": None,
        },
        "shapes": {
            "nb_class": nb_class,
            "input_size": nb_dim,
            "total_size": len(blobs),
            "train_size": len(blobs),
            "test_size": None,
        },
        "batch_formater": lambda x: (x[0], x[1]),
        "label_encoding": None,
        "sample_scale": (0, 1)
    }

def load_scanpy(scanpy_name, condition_field, batch_size, log1p, project_01= True, scanpy_args=None):
    """load dataset from scanpy. """
    import scanpy as sc

    if not scanpy_args:
        adata = getattr(sc.datasets, scanpy_name)()
    else:
        adata = getattr(sc.datasets, scanpy_name)(**scanpy_args)

    if log1p:
        sc.pp.log1p(adata)
    
    if project_01:
        adata.X = adata.X - numpy.min(adata.X)
        adata.X = adata.X / numpy.max(adata.X)

    return make_single_cell_dataset(batch_size, condition_field, adata, scanpy_name)