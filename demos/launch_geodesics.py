from __future__ import print_function, division
import math
import numpy as np
import torch
import torchvision

# Misc
from tqdm import tqdm

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Local imports
import fiberedae.models.fae          as mod
import fiberedae.utils.basic_trainer as vtrain
import fiberedae.utils.persistence   as vpers
import fiberedae.utils.nn            as vnnutils
import fiberedae.utils.plots         as vplots
import fiberedae.utils.fae           as vfae
import fiberedae.utils.datasets      as vdatasets
import fiberedae.utils.useful        as us

# UNUSED CODE FOR NOW
# Compute metric (very slow)
def compute_metric( latent_variable ):
    dimension = latent_variable.shape[0]
    # Tensoring
    latent_variable = latent_variable.view( 1, dimension)
    latent_variable = torch.tensor( latent_variable, requires_grad = True)
    # Compute output
    generated_image = model.forward_decode( encoding = latent_variable )
    generated_image = generated_image.view( generated_image.shape[1] )
    # Compute gradient
    pixel_count = generated_image.shape[0]
    grad = torch.zeros( dimension, pixel_count )
    for j in tqdm(range(pixel_count)):
        generated_image[j].backward( retain_graph=True )
        grad[:,j] = latent_variable.grad
    # Compute metric tensor
    g = torch.mm( grad, torch.t(grad) )
    return g

#----------------------------------------------------------------
# Make grid/list of initial fiber points for computing geodesics
# 
# Note: This function is a bit complicated, as we need to 
#   account for all the use-cases in sampling the latent space:
#   - Random sampling (2d fiber)
#   - Uniform grid (1d fiber)
#   - Uniform grid (2d fiber)
#   - Datasets and empirical sampling

def make_fiber_grid(model, config):

    fiber_grid = []

    if config["sample_mode"]["name"] == "random":
        # Random mode
        for i in range(config["sample_mode"]["geodesics_count"]):
            point = 2*torch.rand(2) - 1
            entry = {
                "src_fiber"     : point, # Fiber coordinate of source
                "src_condition" : config["sample_mode"]["src_condition"],
                "dst_fiber"     : point, # Fiber coordinate of destination
                "src_condition" : config["sample_mode"]["dst_condition"],
                "neighbors"     : [], 
                "curves"        : None
            }
            fiber_grid.append( entry )
    elif config["sample_mode"]["name"] == "grid_1d":
        # One dimensional grid mode
        nb_ticks   = config["sample_mode"]["geodesics_count"]
        grid_step  = 2.0/(nb_ticks+1)
        fiber_grid = [None]*nb_ticks
        for i in range(nb_ticks):
            fiber_grid[i] = [None]*nb_ticks
            x = (i+1)*grid_step - 1.0
            point = torch.tensor( [x] )
            entry = {
                "src_fiber"     : point, # Fiber coordinate of source
                "src_condition" : config["sample_mode"]["src_condition"],
                "dst_fiber"     : point, # Fiber coordinate of destination
                "dst_condition" : config["sample_mode"]["dst_condition"],
                "neighbors"     : [],
                "curves"        : None
            }
            if not i == 0:
                entry["neighbors"].append( i-1 )
            if not i == nb_ticks-1:
                entry["neighbors"].append( i+1 )
            fiber_grid[i] = entry
        # Replace neighbor indices par actual references to grid points
        for i in range(nb_ticks):
            entry = fiber_grid[i]
            neighbors_ref = [ fiber_grid[n] for n in entry["neighbors"] ]
            entry["neighbors"] = neighbors_ref
    elif config["sample_mode"]["name"] == "grid":
        # Grid mode
        nb_ticks   = math.floor( math.sqrt( config["sample_mode"]["geodesics_count"] ) )
        grid_step  = 2.0/(nb_ticks+1)
        fiber_grid = [None]*nb_ticks
        for i in range(nb_ticks):
            fiber_grid[i] = [None]*nb_ticks
            for j in range(nb_ticks):
                x = (i+1)*grid_step - 1.0
                y = (j+1)*grid_step - 1.0
                point = torch.tensor( [x, y] )
                entry = {
                    "src_fiber"     : point, # Fiber coordinate of source
                    "src_condition" : config["sample_mode"]["src_condition"],
                    "dst_fiber"     : point, # Fiber coordinate of destination
                    "dst_condition" : config["sample_mode"]["dst_condition"],
                    "neighbors"     : [],
                    "curves"        : None
                }
                if not i == 0:
                    entry["neighbors"].append( (i-1, j  ) )
                if not i == nb_ticks-1:
                    entry["neighbors"].append( (i+1, j  ) )
                if not j == 0:
                    entry["neighbors"].append( (i  , j-1) )
                if not j == nb_ticks-1:
                    entry["neighbors"].append( (i  , j+1) )
                fiber_grid[i][j] = entry
        # Replace neighbor indices par actual references to grid points
        for i in range(nb_ticks):
            for j in range(nb_ticks):
                entry = fiber_grid[i][j]
                neighbors_ref = [ fiber_grid[n[0]][n[1]] for n in entry["neighbors"] ]
                entry["neighbors"] = neighbors_ref
        # Replace nested list by simple list
        fiber_grid = [
            f for dummy in fiber_grid
                for f in dummy
        ]
    elif config["sample_mode"]["name"] == "dataset":
        if config["dataset"]["name"] == "cufsf":
            # Load dataset
            dataset = us.load_dataset({
                "dataset": config["dataset"],
                "hps":     {"minibatch_size": 256}
            })
            print("")
            train_dataset = dataset["datasets"]["train"]
            # Replace loader with non-shuffling loader
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=False, num_workers=8)
            # Get latent space
            latent_space = vhna.get_latent_space( model, train_loader, batch_formater=dataset["batch_formater"], label_encoding=dataset["label_encoding"])
            latent_space = {
                "base" : latent_space["condition"],
                "fiber": latent_space["observer"],
                "conditions": latent_space["labels"]
            }
            for i in range( config["sample_mode"]["geodesics_count"] ):
                entry = {
                    "src_fiber"     : torch.tensor( latent_space["fiber"][2*i  ] ), # Fiber coordinate of source
                    "src_condition" : 0,
                    "dst_fiber"     : torch.tensor( latent_space["fiber"][2*i+1] ), # Fiber coordinate of destination
                    "dst_condition" : 1,
                    "neighbors"     : [],
                    "curves"        : None
                }
                fiber_grid.append( entry )
        elif config["dataset"]["name"] in ["pancreas", "kang_pbmc", "haber", "menon_retina"]:
            import scanpy as sc
            from scipy.sparse.csr import csr_matrix
            import fiberedae.utils.single_cell as vsc
            from anndata import AnnData
            print("Loading dataset %s..."%config["dataset"]["name"])
            # Loading train
            train = sc.read( config["dataset"]["filepath"], backup_url=config["dataset"]["backup_url"])
            #train.obs["cell_type"] = train.obs["celltype"].tolist()
            train.X = train.X - np.min(train.X)
            train.X = train.X / np.max(train.X)
            # Making dataset into torch DataLoader
            dataset = vdatasets.make_single_cell_dataset(
                batch_size=1024,
                condition_field=config["dataset"]["condition_field"],
                adata=train,
                dataset_name=config["dataset"]["name"],
                oversample=True
            )
            print("")
            # Runs dataset through neural network
            adata             = train                     # train of dataset. adata Has a particular structure specified in scgen.
            condition_key     = config["dataset"]["condition_field"]
            condition_encoder = dataset["label_encoding"]
            batch_size        = 1024
            fiber_points = []
            conditions   = []
            for start in range(0, train.X.shape[0], batch_size):
                stop    = start + batch_size
                samples = adata.X[start:stop]
                conds   = condition_encoder.transform( adata.obs[condition_key][start:stop] )

                # Shameful fix for dataset menon_retina
                if isinstance(samples, csr_matrix):
                    samples = samples.toarray()

                samples = torch.tensor(samples, dtype=torch.float)
                samples = samples.to(model.run_device)
                
                fiber_points[start:stop] = model.observer( samples ).detach().cpu().numpy()
                conditions[start:stop]   = conds
            # Build fiber_grid
            ref_condition = config["sample_mode"]["ref_condition"]
            ref_base      = model.conditions( torch.tensor(ref_condition) ).detach().cpu().numpy()
            for index in range( len(fiber_points) ):
                entry = {
                    "src_fiber"     : fiber_points[index],
                    "src_condition" : conditions[index],
                    "dst_fiber"     : fiber_points[index],
                    "dst_condition" : ref_condition,
                    "neighbors"     : [],
                    "curves"        : None
                }
                fiber_grid.append( entry )
        else:
            raise Exception("Sampling from dataset. Unknown dataset")
    else:
        raise Exception("Unknown sample mode")

    return fiber_grid

# --------------------------------------------------
# Plot functions for each geodesic, when computed
# -- These are for checking convergence
# -- As such, plots are written into disk in file_path

def plot_1d_projections(fiber_dim, curve, file_path):
    plt.rcParams["figure.figsize"] = (20,10)
    dim       = curve.shape[1]
    #
    for i in range(dim):
        plt.subplot(1,dim,i+1)
        plt.plot( range(len(curve[:,i])), curve[:,i] )
        if i<fiber_dim:
            plt.title( "Fiber coord. %d"%(i+1) )
            plt.ylim ( -1, 1)
        else:
            plt.title( "Base coord. %d"%(i+1-fiber_dim) )
    plt.savefig( file_path )
    plt.close()
    return

def plot_energy(naive, full, file_path):
    plt.rcParams["figure.figsize"] = (5,10)
    plt.suptitle( "Evolution of energy during interpolation")
    ax = plt.subplot(1, 1, 1)
    # Compute energy curves
    energy_naive = (naive[1:,:]-naive[:-1,:]).pow(2).sum(dim=1).detach().numpy().cumsum()
    energy_full  = ( full[1:,:]- full[:-1,:]).pow(2).sum(dim=1).detach().numpy().cumsum()
    # Plot
    ax = plt.subplot(1, 1, 1, sharex=ax)
    plt.plot(energy_naive)
    plt.plot(energy_full)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.savefig( file_path )
    plt.close()
    return

def plot_loss(loss, file_path):
    plt.rcParams["figure.figsize"] = (5,10)
    plt.suptitle( "Evolution of energy during gradient descent")
    ax = plt.subplot(1, 1, 1)
    plt.plot(loss)
    plt.ylim( (0,max(loss)) )
    plt.savefig( file_path )
    plt.close()
    return

# --------------------------------------------------------------
# Computes one geodesic
#
# Multithreading friendly
#  - by allowing a reloading of the model from disk

def shooting_iteration(f, c1, c2, counter, geodesics_module, device, epochs, config, model=None):
    """
    f      : Initial fiber point
    c1, c2 : Conditions
    counter: Index of geodesic. Useful for pickling.
    geodesics_module: Needs to be already initialized
    device : For model loading
    config : JSON config
    model  : Model to use. If None, model is loaded from disk.
    """
    import os, pickle
    filename = config["output"] + "geodesic_%05d.p"%counter
    if os.path.exists(filename):
        print("%d. Geodesic already exists!"%counter)
        return
    #
    if c1 == c2:
        return counter, [], []
    fiber_dim = f.shape[0]
    # Set warning level
    import warnings
    warnings.simplefilter("ignore")
    # Reload model, otherwise serialization using Loky is extremely costly
    if not model:
        model, optimizer, history, label_encoding = vpers.load_model( config["model"], mod.FiberedAE, map_location=device)
    # Compute
    b1 = model.conditions( torch.tensor( c1 ) )
    b2 = model.conditions( torch.tensor( c2 ) )
    naive_curve, geodesic_curve, energy = geodesics_module.computeGeodesicShooting( model.forward_decode, b1, b2, f, f*0, epochs=epochs, optimizer_info=config["optimizer"], display_info=str(counter+1) )
    output = (counter, naive_curve, geodesic_curve)
    pickle.dump( output, open(filename, "wb") )
    # Plot 1d projections of geodesics
    if config["plot_options"]["geodesics_1d_projections"]:
        plot_1d_projections( fiber_dim, geodesic_curve, config["output"] + "geodesic_%05d_1d_projections.png"%counter)
    # Curve of generated samples
    naive = model.forward_decode( encoding = torch.tensor( naive_curve ) )
    full  = model.forward_decode( encoding = torch.tensor( geodesic_curve  ) )
    # Plot energy evolution
    plot_energy( naive, full, config["output"] + "geodesic_%05d_energy_evolution.png"%counter )
    # Plot loss
    plot_loss( energy, config["output"] + "geodesic_%05d_loss.png"%counter )
    return output
# End shooting_iteration

#---------------------------------------
# Global plot functions 

def plot_fiber_manifolds( model, condition1, condition2):
    plt.rcParams["figure.figsize"] = (20,10)
    plt.subplot(121)
    F1 = vplots.latent_grid_plot(
        model=model,
        condition=condition1,
        nb_ticks=12,
        start=0,
        stop=2*np.pi
    )
    plt.title( "Condition1" )
    plt.subplot(122)
    F2 = vplots.latent_grid_plot(
        model=model,
        condition=condition2,
        nb_ticks=12,
        start=0,
        stop=2*np.pi
    )
    plt.title( "Condition2" )
    return plt

def plot_geodesics_1d_projections( time_grid, fiber_grid):
    plt.rcParams["figure.figsize"] = (20,10)
    for datum in fiber_grid:
        curve     = datum["curves"]["full"]
        fiber_dim = datum["src_fiber"].shape[0]
        dim       = curve.shape[1]
        #
        for i in range(dim):
            plt.subplot(1,dim,i+1)
            plt.plot( time_grid, curve[:,i] )
            if i<fiber_dim:
                plt.title( "Fiber coord. %d"%(i+1) )
                plt.ylim ( -1, 1)
            else:
                plt.title( "Base coord. %d"%(i+1-fiber_dim) )
    return plt

def plot_diffeomorphism( fiber_grid ):
    # - Prepare data
    source_vertices      = [ datum["src_fiber"] for datum in fiber_grid ]
    destination_vertices = [ datum["dst_fiber"] for datum in fiber_grid ]
    destination_edges    = [ (f["dst_fiber"], n["dst_fiber"])
        for f in fiber_grid
            for n in f["neighbors"]
    ]
    # - Finish with matplotlib
    import matplotlib
    line_collection = matplotlib.collections.LineCollection(destination_edges, linewidths=2)
    plt.rcParams["figure.figsize"] = (20,10)
    fig, ax = plt.subplots()
    ax.add_collection( line_collection )
    plt.scatter( [v[0] for v in destination_vertices], [v[1] for v in destination_vertices] )
    plt.scatter( [v[0] for v in source_vertices]     , [v[1] for v in source_vertices] )
    plt.title("Diffeomorphism between fibers")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    return plt

def plot_geodesics_interpolation(model, fiber_grid, frames):
    def plot_interpolation(flag, frames):
        count    = len(fiber_grid)
        result   = []
        for i in range(count):
            datum = fiber_grid[i]
            curve = torch.tensor( datum["curves"][flag] )
            # Generate images
            decoding       = model.forward_decode( encoding = curve )
            # Save
            datum["curves"][flag+"_decoding"] = decoding
            # Subsample
            time_steps     = int( decoding.shape[0] )
            decoding       = decoding[range(0, time_steps, math.floor(time_steps/frames) ),:]
            # Formatting
            pixels_count   = decoding.shape[1]
            pixels_per_dim = math.floor( math.sqrt( pixels_count ) )
            decoding       = decoding.view( decoding.shape[0], 1, pixels_per_dim, pixels_per_dim )
            # Makegrid and append
            imgs  = torchvision.utils.make_grid( decoding, nrow=frames)
            result.append(imgs) 
        return result
    plt.suptitle( "Naive (top) vs geodesic (bottom) interpolation" )
    naive_interpolations    = plot_interpolation( flag="naive", frames=frames)
    geodesic_interpolations = plot_interpolation( flag="full" , frames=frames)
    img_count = len(naive_interpolations)
    plt.axis('off')
    for i in range( img_count ):
        plt.subplot(img_count, 1, i+1)
        imgs1 = naive_interpolations[i]
        imgs2 = geodesic_interpolations[i]
        imgs  = torch.cat( (imgs1, imgs2), 1)
        # From torch to numpy
        npimg = imgs.detach().cpu().numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        # Plot
        plt.imshow( npimg )
    return plt

def plot_energy_evolutions(model, fiber_grid):
    count = len(fiber_grid)
    plt.rcParams["figure.figsize"] = (5,10)
    plt.suptitle( "Evolution of energy during interpolation")
    ax = plt.subplot(count, 1, 1)
    for i in range(count):
        datum = fiber_grid[i]
        # Curve of generated samples
        naive = model.forward_decode( encoding = torch.tensor( datum["curves"]["naive"] ) )
        full  = model.forward_decode( encoding = torch.tensor( datum["curves"]["full"]  ) )
        # Compute energy curves
        energy_naive = (naive[1:,:]-naive[:-1,:]).pow(2).sum(dim=1).detach().numpy().cumsum()
        energy_full  = ( full[1:,:]- full[:-1,:]).pow(2).sum(dim=1).detach().numpy().cumsum()
        # Plot
        ax = plt.subplot(count, 1, i+1, sharex=ax)
        plt.plot(energy_naive)
        plt.plot(energy_full)
        plt.setp(ax.get_xticklabels(), visible=False)
    return plt

def plot_geodesics_correspondence( model, fiber_grid, nb_ticks ):
    src_fiber = torch.tensor( [ datum["src_fiber"] for datum in fiber_grid ] )
    dst_fiber = torch.tensor( [ datum["dst_fiber"] for datum in fiber_grid ] )
    src_base  = torch.tensor( [ datum["src_base"]  for datum in fiber_grid ] )
    dst_base  = torch.tensor( [ datum["dst_base"]  for datum in fiber_grid ] )
    initial_latent_variable = torch.cat( ( src_fiber, src_base), 1 )
    naive_latent_variable   = torch.cat( ( src_fiber, dst_base), 1 )
    final_latent_variable   = torch.cat( ( dst_fiber, dst_base), 1 )
    src_decoding   = model.forward_decode( encoding = initial_latent_variable )
    naive_decoding = model.forward_decode( encoding = naive_latent_variable )
    geode_decoding = model.forward_decode( encoding = final_latent_variable )
    # Plots
    decodings = [
        src_decoding,
        naive_decoding,
        geode_decoding
    ]
    decoding_strings = [
        "Src. decoding",
        "Naive decoding",
        "Geodesic decoding"
    ]
    for i in range( len(decodings)):
        pixels_count   = decodings[i].shape[1]
        pixels_per_dim = math.floor( math.sqrt( pixels_count ) )
        decodings[i]  = decodings[i].view( len(fiber_grid), 1, pixels_per_dim, pixels_per_dim )
    for i in range( len(decodings)):
        plt.subplot(1, 3, i+1)
        imgs  = torchvision.utils.make_grid( decodings[i], nrow=nb_ticks)
        npimg = imgs.detach().cpu().numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        plt.title( decoding_strings[i] )
        plt.imshow(npimg)
    return plt

def plot_geodesics_3d(time_grid, fiber_grid):
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    for datum in fiber_grid:
        curve = datum["curves"]["fiber"]
        ax.plot( curve[:,0], curve[:, 1], time_grid, label='parametric curve')
    ax.legend()
    return plt

if __name__ == '__main__':
    import argparse
    import os
    import torch
    import time
    import json

    start_time = time.ctime()

    parser = argparse.ArgumentParser(description='Perform geodesic transport between fibers for a sample')
    parser.add_argument("--json-config"    , metavar="file"  , help="Configuration file" , type=str, action="store", default="./geodesics_images/mnist.json")
    parser.add_argument("--epochs"         , metavar="number", help="Number of epochs for gradient descent", type=int, action="store", default="400")
    parser.add_argument("--device"         , help="device on which to run", type=str, action="store", default="cpu")
    parser.add_argument("--multithreaded"  , help="computes geodesics using joblib Parallel", action="store_true", default=False)
    
    args = parser.parse_args()
    args = args.__dict__
 
    print("")

    print("Loading json...")
    config  = json.load(open(args["json_config"]))

    print("Loading model %s..." % config["model"])
    model, optimizer, history, label_encoding = vpers.load_model( config["model"], mod.HNA_WithCond, map_location=args['device'])

    label_encoder, label_decoder = label_encoding.transform, label_encoding.inverse_transform
    print("")

    # Manifold plots: Plots fibers above the two conditions
    if config["plot_options"]["latent_grid"]:
        plot = plot_fiber_manifolds( model, 
            condition1 = config["sample_mode"]["src_condition"],
            condition2 = config["sample_mode"]["dst_condition"] )
        plot.savefig( "./geodesics_images/save_conditions.png" )
        plot.close()

    # Prepare grid/list of initial points (f,b) depending on the JSON config file
    fiber_grid = make_fiber_grid(model, config)

    # Pickling job information in output file
    import os, pickle
    total_count   = len(fiber_grid)
    compute_count = np.sum( [ x["src_condition"] != x["dst_condition"] for x in fiber_grid ] )
    print( "Pickling job information: %d / %d geodesics to compute..."%(compute_count, total_count) )
    if not os.path.exists( config["output"] ):
        os.makedirs( config["output"], exist_ok=True)
    try:
        pickle.dump( fiber_grid, open( config["output"] + "fiber_grid_nojob.p", "wb+" ) )
    except Exception as error:
        print("Could not pickle:", error)

    #------------------------------------------
    # Initialize module for computing geodesics
    from fiberedae.models.geometry import NumericalGeodesics
    geodesics_module = NumericalGeodesics( n_max=config["numerics"]["n_max"], step_count=config["numerics"]["step_count"])

    # Loop over initial fiber points and compute geodesics
    geodesics = [None for _ in fiber_grid]
    if (config["geodesic_mode"]=="shooting") & args["multithreaded"]:
        print( "Computing geodesics in shooting mode, multithreaded...")
        from joblib import Parallel, delayed
        Parallel(n_jobs=-1)( delayed(shooting_iteration)(
            fiber_grid[counter]["src_fiber"], 
            fiber_grid[counter]["src_condition"],
            fiber_grid[counter]["dst_condition"],
            counter,
            geodesics_module=geodesics_module,
            device=args["device"],
            epochs=args["epochs"], 
            config=config,
            model=None) for counter in range(len(fiber_grid)))
    elif config["geodesic_mode"]=="shooting":
        print("Computing geodesics in shooting mode, single-threaded...")
        for counter in range( len(fiber_grid) ):
            datum     = fiber_grid[counter]
            shooting_iteration( 
                datum["src_fiber"], 
                datum["src_condition"], 
                datum["dst_condition"], 
                counter, 
                geodesics_module=geodesics_module,
                device=args["device"],
                epochs=args["epochs"], 
                config=config,
                model=model)
        # End for
    elif config["geodesic_mode"]=="interpolation":
        for index in range( len(fiber_grid) ):
            datum     = fiber_grid[index]
            f1        = datum["source"]
            f2        = datum["destination"]
            fiber_dim = f1.shape[0]
            # Only in this example we want f1 = f2
            m1  = torch.cat( (f1, b1), -1 )
            m2  = torch.cat( (f2, b2), -1 )
            naive_curve, geodesic_curve = geodesics_module.computeGeodesicInterpolation( model.forward_decode, m1, m2, epochs=args["epochs"], optimizer_info=config["optimizer"], display_info=str(index+1) )
            geodesics.append( (index, naive_curve, geodesic_curve) )
        # End for
    # End of case disjunctions of the different modes
    
    # Load all geodesics from disc
    print("Loading all geodesics from disc and integration into a single data structure...")
    for index in range(len(fiber_grid)):
        datum = fiber_grid[index]
        if datum["src_condition"] == datum["dst_condition"]:
            base_point     = model.conditions( torch.tensor( datum["dst_condition"] ) ).detach().cpu().numpy()
            point          = np.concatenate( (datum["src_fiber"], base_point) )
            constant_curve = np.ones( (config["numerics"]["step_count"], 1))*point
            geodesic       = index, constant_curve, constant_curve
        else:
            filename = config["output"] + "geodesic_%05d.p"%index
            stream   = open(filename, "rb")
            geodesic = pickle.load( stream )
            assert(geodesic[0]==index)
            stream.close()
        geodesics[index] = geodesic
    # End for
    #
    print("Cleanup and saving on disk in fiber_grid.p and fiber_grid_endpoints.p ...")
    print("")
    geodesics_endpoints = []
    for (index, naive_curve, geodesic_curve) in geodesics:
        datum     = fiber_grid[index]
        f         = datum["src_fiber"]
        fiber_dim = f.shape[0]
        # Separate geodesic curve into fiber, base, naive and full
        curves = {
            "fiber": geodesic_curve[:, :fiber_dim],
            "base" : geodesic_curve[:, fiber_dim:],
            "full" : geodesic_curve,
            "naive": naive_curve
        }
        datum["curves"]    = curves
        datum["src"]       = curves["full"][ 0,:]
        datum["dst"]       = curves["full"][-1,:]
        datum["src_fiber"] = curves["fiber"][ 0,:]
        datum["dst_fiber"] = curves["fiber"][-1,:]
        datum["src_base" ] = curves["base" ][ 0,:]
        datum["dst_base" ] = curves["base" ][-1,:]
        # Save endpoints separately
        endpoints = {}
        for key in datum:
            if key in ["curves", "neighbors"]:
                continue
            endpoints[key] = datum[key]
        geodesics_endpoints.append( endpoints )
    # End for
    try:
        pickle.dump( fiber_grid         , open( config["output"] + "fiber_grid.p", "wb+" ) )
        pickle.dump( geodesics_endpoints, open( config["output"] + "fiber_grid_endpoints.p", "wb+" ) )
    except Exception as error:
        print("Could not pickle:", error)

    # -----------------------------------------
    # Global plots

    print("Plots:")

    #Plot 1d projections of geodesics
    if config["plot_options"]["geodesics_1d_projections"]:
        print("- Plotting 1d projections of geodesics...")
        time_grid = geodesics_module.time_grid.detach().numpy()
        plot_geodesics_1d_projections( time_grid, fiber_grid )
        plot.savefig( "./geodesics_images/save_geodesics_1d_projections.png" )
        plot.close()

    # Grid plot in 2D of diffeomorphism (In geodesic shooting mode only)
    if (config["geodesic_mode"]=="shooting") & config["plot_options"]["geodesics_diffeomorphism"]:
        print("- Plotting 2D diffeomorphism...")
        plot = plot_diffeomorphism( fiber_grid )
        plot.savefig( "./geodesics_images/save_geodesics_diffeo.png" )
        plot.close()

    # Interpolation along geodesics the neural network output
    if config["plot_options"]["geodesics_interpolation"]:
        print( "- Plotting naive vs geodesic interpolations..." )
        plot = plot_geodesics_interpolation( model, fiber_grid, frames=16)
        plot.savefig( "./geodesics_images/save_geodesics_interpolations.png" )
        plot.close()
    
    # Energy curves
    if config["plot_options"]["geodesics_energy_curves"]:
        print( "- Plotting energy evolutions...")
        plot = plot_energy_evolutions(model, fiber_grid)
        plot.savefig( "./geodesics_images/save_geodesics_energy_evolution.png" )
        plot.close()
    
    # Correspondence for the neural network output
    if config["plot_options"]["geodesics_correspondence"]:
        print( "- Plotting correspondence...")
        count     = config["sample_mode"]["geodesics_count"]
        nb_ticks  = math.floor( math.sqrt( count ) )
        plot      = plot_geodesics_correspondence( model, fiber_grid, nb_ticks )
        plot.savefig( "./geodesics_images/save_geodesics_correspondence.png" )
        plot.close()

    # Plot 3D of geodesics
    if config["plot_options"]["geodesics_3d"]:
        print( "- Plotting geodesics in 3d...")
        plot = plot_geodesics_3d( time_grid, fiber_grid )
        plot.show()
    
    # Done
    print("done\n")
