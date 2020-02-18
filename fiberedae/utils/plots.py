import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cm

import torchvision
import torch
import numpy

def tidyfy(labels, **other):
    import numpy
    labels = numpy.asarray(labels)
    unique_labels = numpy.unique(labels)
    
    dct_filter = other
    res = { key:[] for key in dct_filter }
    res["labels"] = []
    
    for lab in unique_labels:
        idx = labels == lab
        res["labels"].append(lab)
        for key in other.keys():
            res[key].append(dct_filter[key][idx])
    
    return res

def scatter_cluster_density(
        x,
        y,
        labels,
        title="",
        x_label="",
        y_label="",
        point_opacity=0.2,
        point_size=2,
        fig_height=10,
        fig_width=10,
        color_map=cm.rainbow,
        nb_bins=100,
        stacked_bins=True,
        force_subsampling=None,
        hist_size=0.15,
        x_range=None,
        y_range=None,
    ) :
 
    if x_range is None:
        margin = (numpy.max(x) - numpy.min(x)) / 100
        x_range = (numpy.min(x)-margin, numpy.max(x)+margin)
    
    if y_range is None:
        margin = (numpy.max(y) - numpy.min(y)) / 100
        y_range = (numpy.min(y)-margin, numpy.max(y)+margin)

    plot_dct = tidyfy(labels, x=x, y=y)
    
    nullfmt = NullFormatter()         # no labels

    # definitions for the axe_
    ratio = 0.65
    left, width = hist_size, ratio
    bottom, height = hist_size, ratio
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, hist_size]
    rect_histy = [left_h, bottom, hist_size, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(fig_height, fig_width))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot
    for xv, yv, labels in zip(plot_dct["x"], plot_dct["y"], plot_dct["labels"]) :
        if force_subsampling and force_subsampling < len(xv):
            xv = numpy.random.choice(xv, force_subsampling, replace=False)
            yv = numpy.random.choice(yv, force_subsampling, replace=False)
        axScatter.scatter(xv, yv, label = labels, alpha=point_opacity, s=point_size)
    axScatter.legend()
    
    # now determine nice limits by hand:
    xmax = numpy.max(numpy.fabs(x)) 
    ymax = numpy.max(numpy.fabs(y))
    xymax = numpy.max([xmax, ymax])
    
    axScatter.set_xlim(x_range)
    axScatter.set_ylim(y_range)

    xbins = numpy.arange(numpy.min(x), numpy.max(x), (numpy.max(x)-numpy.min(x)) / nb_bins)
    ybins = numpy.arange(numpy.min(y), numpy.max(y), (numpy.max(y)-numpy.min(y)) / nb_bins)
    axHistx.hist(plot_dct["x"], bins=xbins, stacked=stacked_bins)
    axHisty.hist(plot_dct["y"], bins=ybins, orientation='horizontal', stacked=stacked_bins)

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.suptitle(title, fontsize=20)
    axScatter.set_xlabel(x_label, fontsize=18)
    axScatter.set_ylabel(y_label, fontsize=16)
    
    return plt

def plot_latent(vals, labels, point_opacity=0.2, fig_height=10, fig_width=10):
    import umap

    if vals.shape[1] == 1:
        fig = scatter_cluster_density(
            vals[:, 0],
            vals[:, 0],
            labels = labels,
            title="",
            x_label="D1",
            y_label="D2",
            fig_height=fig_height,
            fig_width=fig_width,
            point_opacity=point_opacity
        )
    elif vals.shape[1] == 2:
        fig = scatter_cluster_density(
                vals[:, 0],
                vals[:, 1],
                labels = labels,
                title="",
                x_label="D1",
                y_label="D2",
                fig_height=fig_height,
                fig_width=fig_width,
                point_opacity=point_opacity
            )
    elif vals.shape[1] >= 2:
        if vals.shape[1] > 2 :
            reducer = umap.UMAP()
            vals = reducer.fit_transform(vals)
        fig = scatter_cluster_density(
            vals[:, 0],
            vals[:, 1],
            labels = labels,
            title="",
            x_label="UMAP1",
            y_label="UMAP2",
            fig_height=fig_height,
            fig_width=fig_width,
            point_opacity=point_opacity
        )

    return fig

def plot_fiber_plus_cond(xy_fiber, xy_cond, labels, label_decoder, scale_factor=1):
    # plt.figure(figsize=(8, 6))
    unique_labels = numpy.unique(labels)
    nb_class = len(unique_labels)

    v1 = (xy_fiber[:, 0] + xy_cond[:, 0]*scale_factor)
    v2 = (xy_fiber[:, 1] + xy_cond[:, 1]*scale_factor)
    
    fig, ax = plt.subplots()
    for colid, g in enumerate( unique_labels ):
        ix = numpy.where(labels == g)
        ax.scatter(v1[ix], v2[ix], label = label_decoder([g])[0])
    ax.legend()
    return plt

def get_latent_ticks(model, nb_ticks, start, stop, diagonal=False):
    ticks = numpy.linspace(start, stop, nb_ticks)
    ticks = [ticks] * model.fiber.out_dim
    if diagonal :
        grid = numpy.array(ticks).transpose()
    else :
        grid = numpy.meshgrid(*ticks)
        grid = numpy.array(grid).transpose()
        grid = grid.reshape((-1, model.fiber_dim))

    grid = torch.tensor(grid, dtype=torch.float ).to(model.run_device)

    if model.fiber.projection_type == "cube":
        fiber_values = torch.sin(grid)
    elif model.fiber.projection_type == "torus" :
        half = grid.shape[1]//2
        sin, cos = grid[:, :half], grid[:, half:]
        fiber_values = torch.cat([torch.sin(sin), torch.cos(cos)], 1 )
    else:
        raise ValueError("Invalid model.fiber.projection_type '%s', not int 'cube', ',torus'" % (model.fiber.projection_type))
    
    return fiber_values

def get_latent_grid(model, condition, nb_ticks, max_normalize=True, scaling=None, start=0, stop=2*numpy.pi):
    if model.fiber_dim > 2:
        tmp_nb_ticks = int(nb_ticks / model.fiber_dim)
        if nb_ticks > 0:
            nb_ticks = tmp_nb_ticks
    
    fiber_values = get_latent_ticks(model, nb_ticks, start, stop, diagonal=False)
    conditions = [condition] * fiber_values.shape[0]
    conditions = torch.tensor( numpy.array(conditions).reshape(-1, 1)).to(model.run_device)

    imgs = model.forward_output(x=fiber_values, cond=conditions, fiber_input=True)
    imgs = imgs.detach()

    if max_normalize:
        maxes = torch.max(imgs)
        imgs = imgs / maxes 
    if scaling:
        imgs = imgs * scaling

    side_size = int(numpy.sqrt(imgs.shape[1]))
    # print(imgs.shape, fiber_values.shape)
    try :
        imgs = imgs.view((-1, 1, side_size, side_size))
    except :
        imgs = imgs.view((-1, 3, side_size, side_size)) #rgb
    
    return imgs, fiber_values

def latent_grid_plot(model, condition, nb_ticks, max_normalize=True, scaling=None, start=0, stop=2*numpy.pi, plot_inputs=False, grid_height=20, grid_width=20):
    import matplotlib.pyplot as plt

    plt.rcParams['figure.figsize'] = [grid_height, grid_width]
    
    plt.figure(1)
    
    imgs, ticks = get_latent_grid(model, condition, nb_ticks, max_normalize, scaling, start, stop)
    ticks = ticks.detach().cpu().numpy()
    imgs = torchvision.utils.make_grid(imgs, nrow=nb_ticks)
    npimg = imgs.detach().cpu().numpy()
    npimg = numpy.transpose(npimg, (1, 2, 0))
    if plot_inputs:
        plt.subplot(2, 1, 1)
    plt.imshow(npimg)

    if plot_inputs:
        plt.subplot(2, 1, 2)
        
        size = ticks.shape[1]
        if model.fiber.projection_type == "torus":
            half = ticks.shape[1]//2
            sin, cos = ticks[:, :half], ticks[:, half:]
            f = 1
            for dim in range(sin.shape[1]):
                plt.plot(f*sin[:,dim], f*cos[:,dim], "o--")
                f = f + 0.1
        else :
            for dim in range(size):
                y = ticks[:,dim]
                x = range(len(y))
                plt.plot(x, y)

    return plt

def save_latent_gif(model, condition, nb_ticks, filename, max_normalize=True, scaling=None, start=None, stop=None):
    import imageio

    imgs, ticks = get_latent_grid(model, condition, nb_ticks, max_normalize, scaling, start, stop)
    npimg = imgs.detach().cpu().numpy()
    
    s = npimg.shape
    npimg = numpy.reshape(npimg, (s[0], s[2], s[3]))
    
    npimg /= numpy.max(npimg)
    npimg *= 255
    npimg = npimg.astype("uint8")
    imageio.mimsave(filename, npimg)
    
def xy_line_plot(x_values, y_values, x_label, y_label, title):
    plt.figure()
    
    x_vals = numpy.array(x_values)
    y_vals = numpy.array(y_values)
    
    plt.plot(x_vals, y_vals)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    return plt

def line_plot(y_values, x_label, y_label, title, cumulative, sort):
    plt.figure()
    
    x_vals = numpy.arange(len(y_values))
    y_vals = numpy.array(y_values)
    
    if sort : y_vals.sort()
    if cumulative : y_vals = numpy.cumsum(y_vals)

    return xy_line_plot(x_vals, y_vals, x_label, y_label, title)

def interactive_line_plots(dct_values, x_label, y_label, height=600, width=1200, sub_sets = None, nb_cols=1, extension="bokeh"):
    """works with dicts label->values or with a list of dicts"""
    def _overlay(dct):
        elmt = None
        for label, values in dct.items():
            if elmt is None :
                elmt = hv.Curve(values, x_label, y_label, label=label)
            else :
                elmt = elmt * hv.Curve(values, x_label, y_label, label=label)
        return elmt.opts(width=width, height=height)
    
    import holoviews as hv
    hv.extension(extension)
    
    if type(dct_values) is dict:
        elmt = _overlay(dct_values)
    else :
        elmt = _overlay(dct_values[0])
        for dct in dct_values[1:]:
            elmt += _overlay(dct)
        elmt = elmt.cols(nb_cols)

    return elmt