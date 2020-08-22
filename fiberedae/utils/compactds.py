import os
import pickle
import scanpy as sc
import numpy as np
from sklearn import preprocessing
import scipy.sparse

class CompactDS(object):
    """docstring for CompactDS"""
    def __init__(self, X, Y, label_encoder):
        super(CompactDS, self).__init__()
        self.X = X
        self.Y = Y
        self.label_encoder = label_encoder

    def densify(self):
        if scipy.sparse.issparse(self.X):
            self.X = self.X.todense()

def anndata_to_compact(dataset, x_key, y_key, output_folder):
    """
    Make a memory efficient version of the anndata
    """

    if dataset.endswith(".h5"):
        adata = sc.read_10x_h5(dataset)
    elif dataset.endswith(".h5ad"):
        adata = sc.read(dataset)
    else:
        raise ValueError("Extension must be either h5 or h5ad")

    if x_key == "X":
        X = adata.X
    elif x_key in adata.obsm :
        X = adata.obsm[x_key]
    elif x_key in adata.obs :
        X = adata.obs[x_key]
    else:
        raise KeyError("%s is not present in obsm or obs" % x_key)

    X = X.astype(dtype="float32")

    if y_key is not None :
        if y_key in adata.obsm :
            Y = adata.obsm[y_key]
        elif y_key in adata.obs :
            Y = adata.obs[y_key]
        else:
            raise KeyError("%s is not present in obsm or obs" % y_key)

        le = preprocessing.LabelEncoder()
        le.fit(Y)
        Y = le.transform(Y)
        if np.max(Y) < 32767:
            Y = Y.astype(dtype="int16")

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    print("saving...")
    
    print("\tX...")
    filename=os.path.join(output_folder, "X")
    scipy.sparse.save_npz(filename, X)
    
    if y_key is not None :
        print("\tY...")
        filename=os.path.join(output_folder, "Y")
        np.save(filename, Y)

        print("\tlabel encoder...")
        filename=os.path.join(output_folder, "label_encoder.pkl")
        with open(filename, "wb") as f :
            pickle.dump(le, f)
    
    print("done.")

def read(folder):
    X = scipy.sparse.load_npz(os.path.join(folder, "X.npz"))
    
    Y, le = None, lambda x: 0
    y_filename = os.path.join(folder, "Y.npy")
    if os.path.exists(y_filename):
        Y = np.load(y_filename)
        with open(os.path.join(folder, "label_encoder.pkl"), "rb") as f:
            le = pickle.load(f)

    return CompactDS(X, Y, le)
