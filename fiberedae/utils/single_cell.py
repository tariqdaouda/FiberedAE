import os
import numpy
import tarfile
import tempfile

import scanpy as sc
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def load_10x_dataset(filepath, backup_url=None):
    """load an h5ad (v3) of an mtx in a tar.gz (v1, v2) file"""
    
    filename = os.path.basename(filepath)
    adata = sc.read(filepath, backup_url=backup_url)

    adata.obs["dataset"] = pd.Series()
    adata.obs["dataset"] = filename
    
    adata.X = adata.X - numpy.min(adata.X)
    adata.X = adata.X / numpy.max(adata.X)

    return adata

def translate(model, adata, condition_key, ref_condition, condition_encoder, batch_size, X_field = None):
    """
    Naive transfer implementation using the network only (No geodesic transport)
    Corrects batch effect by translating data from an anndata object into a reference condition.
    Return a anndata object with adjusted gene expressions
    """
    import torch
    from anndata import AnnData
    from scipy.sparse import issparse
    from tqdm import trange

    if ref_condition:
        ref_condition = condition_encoder.transform( [ref_condition] )[0]
    
    if X_field is None :
        new_x = adata.X.copy()
        if issparse(new_x):
            new_x = new_x.todense()
    else :
        new_x = adata.obsm[X_field].copy()
    
    for start in trange(0, new_x.shape[0], batch_size):
        stop = start + batch_size
        samples = new_x[start:stop]
        conds = condition_encoder.transform( adata.obs[condition_key][start:stop] )
        
        samples = torch.tensor(samples, dtype=torch.float)
        samples = samples.to(model.run_device)

        conds = torch.tensor( conds )
        conds = conds.to(model.run_device)

        if ref_condition is not None:
            conds = conds - conds + ref_condition
        else:
            conds = conds

        out = model.forward_output(x=samples, cond=conds)
        new_x[start:stop] = out.detach().cpu().numpy()

    if X_field is None :
        ret = AnnData(X = new_x, obs=adata.obs, var=adata.var, obsm=adata.obsm, varm=adata.varm, uns=adata.uns )
    else:
        ret = adata.copy()
        ret.obsm[X_field] = new_x
    return ret

class BatchcorrectionEvaluator(object):
    """Metrics for evaluating single cell sequencing batch correction results"""
    def __init__(self, do_cv=False):
        super(BatchcorrectionEvaluator, self).__init__()
        self.do_cv = do_cv
        self.classifiers = ["rf", 'svc', 'logreg']

    def cv_train_and_score_test(self, x, y, base_model, hp_grid, cv=5, seed=42):
        clf = GridSearchCV(base_model, hp_grid, cv=cv, n_jobs=-1)
        x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.25, random_state=seed)
        clf.fit(x_train, y_train)
        return clf.score(x_test, y_test)

    def _get(self, adata, field):
        if field in adata.obs:
            val = adata.obs[field]
        elif field in adata.obsm:
            val = adata.obsm[field]
        else :
            raise ValueError("AnnData object has no field: %s" % field)
        return val

    def calc_ward_variances(self, adata, y_field, X_field=None ):
        def variance( x ):
            count  = x.shape[0]
            dim    = x.shape[1]
            mean   = numpy.mean( x, axis=0).reshape( (1,dim) )
            result = numpy.mean( numpy.linalg.norm(x - numpy.ones( (count,1) )*mean, axis=1)**2 )
            return result
        
        if X_field is None :
            X = adata.X
        else :
            X = self._get(adata, X_field)
        y = self._get(adata, y_field)

        result = {}
        
        count = len(X)
        dim   = len(X[0])
        #
        # Compute total variance
        mean          = numpy.mean( X, axis=0)
        var           = variance(X)
        result["var"] = var
        #
        # Form groups
        group_ids     = list( set( [x for x in y] ) )
        # Indices for each group
        group_indices = [None]*len( group_ids )
        for index in range( len(group_indices) ):
            group_indices[index] = numpy.where( y == group_ids[index] )[0]
        #
        # Mean and intra-group variance for each group
        group_means   = [None]*len( group_ids )
        group_vars    = [None]*len( group_ids )
        group_inertia = [None]*len( group_ids )
        for index in range( len(group_indices) ):
            data = X[ group_indices[index] ]
            m    = numpy.mean( data, axis=0 )
            v    = variance( data )
            i    = numpy.linalg.norm( m - mean )**2
            group_means  [index] = m
            group_vars   [index] = v
            group_inertia[index] = i
        
        # Intra-group and inter-group inertias
        intra_inertia = 0
        inter_inertia = 0
        for index in range( len(group_indices) ):
            group_count   = len( group_indices[index] )
            intra_inertia = intra_inertia + group_vars[index]*group_count/count
            inter_inertia = inter_inertia + group_inertia[index]*group_count/count
        # End for
        #
        result["intra_class"] = intra_inertia
        result["inter_class"] = inter_inertia
        return result
    
    def calc_lisi(self, adata, y_field, X_field=None):
        try :
            import harmonypy as hpy
        except :
            print("Please install harmonypy for lisi score")
            raise
        
        if X_field is None :
            X = adata.X
        else :
            X = self._get(adata, X_field)
        y = self._get(adata, y_field)
    
        lisi = hpy.compute_lisi(X, adata.obs, [y_field])
        vals = lisi[:, 0]
        ret = {"mean": numpy.mean(vals), "std": numpy.std(vals)}
        return ret

    def calc_batch_accuracy(self, adata, y_field, X_field=None, classifier="logreg"):
        """returns mean accuracy of classifier"""
        
        if X_field is None :
            X = adata.X
        else :
            X = self._get(adata, X_field)
        y = self._get(adata, y_field)

        if classifier.lower() == "logreg" :
            clf = LogisticRegression(class_weight="balanced", multi_class='auto', solver="lbfgs")
            parameters = {'C': numpy.logspace(-2, 1, 20)}
        elif classifier.lower() == "svc" :
            clf = SVC(kernel='rbf', class_weight="balanced")
            parameters = {'C': numpy.logspace(-2, 1, 20),
                          'gamma': numpy.logspace(-2, 2, 10).tolist() + ['auto', 'scale']}
        elif classifier.lower() == "rf":
            clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1)
            parameters = {}
        else:
            raise ValueError("Unknown classifier: %s" % classifier)

        if not self.do_cv:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            clf.fit(x_train, y_train)
            return clf.score(x_test, y_test)
        else:
            return self.cv_train_and_score_test(X, y, base_model=clf, hp_grid=parameters, cv=5)

    def calc_kBET(self, adata, y_field, X_field):
        """return the result kbet statistical score"""
        return pg.calc_kBET(adata, attr=y_field, rep=X_field.replace("X_", ""))

    def evaluate(self, adata, y_field, X_field=None):
        """return a dict with results for all evaluation methods"""
        ret = { cl_name: self.calc_batch_accuracy(adata=adata, y_field=y_field, X_field=X_field, classifier=cl_name) for cl_name in self.classifiers }
        ret["lisi"] = self.calc_lisi(adata=adata, y_field=y_field, X_field=X_field)
    
        ret["ward"] = self.calc_ward_variances(adata, y_field, X_field=X_field)
        ret["kbet"] = None
        if X_field is not None :
            try:
                ret["kbet"] = self.calc_kBET(adata, X_field = X_field, y_field=y_field)
            except:
                pass
        return ret

    def pre_post_eval(self, pre_adata, post_adata, X_field, batch_field, cell_field):
        """return a dict with results for all evaluation methods"""
        ret = dict()
        for cl_name in self.classifiers:
            ret['pre_batch_'+cl_name] = self.calc_batch_accuracy(adata=pre_adata, y_field=batch_field, X_field=X_field, classifier=cl_name)
            ret['post_batch_' + cl_name] = self.calc_batch_accuracy(adata=post_adata, y_field=batch_field, X_field=X_field, classifier=cl_name)
            ret['pre_cell_'+ cl_name] = self.calc_batch_accuracy(adata=pre_adata, y_field=cell_field, X_field=X_field, classifier=cl_name)
            ret['post_cell_' + cl_name] = self.calc_batch_accuracy(adata=post_adata, y_field=cell_field, X_field=X_field, classifier=cl_name)

        ret["pre_lisi"] = self.calc_lisi(adata=pre_adata, y_field=batch_field, X_field=X_field)
        ret["post_lisi"] = self.calc_lisi(adata=post_adata, y_field=batch_field, X_field=X_field)
        ret["pre_clisi"] = self.calc_lisi(adata=pre_adata, y_field=cell_field, X_field=X_field)
        ret["post_clisi"] = self.calc_lisi(adata=post_adata, y_field=cell_field, X_field=X_field)

        ret['pre_ward'] = self.calc_ward_variances(pre_adata, batch_field, X_field=X_field)
        ret['post_ward'] = self.calc_ward_variances(post_adata, batch_field, X_field=X_field)
        ret["pre_kbet"] = self.calc_kBET(pre_adata, X_field=X_field, y_field=batch_field)
        ret["post_kbet"] = self.calc_kBET(post_adata, X_field=X_field, y_field=batch_field)

        return ret


