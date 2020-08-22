import torch
import fiberedae.utils.datasets as vdatasets
import fiberedae.utils.basic_trainer as vtrain

def load_dataset(config):
    kwargs = {"batch_size": config["hps"]["minibatch_size"]}
    try:
        kwargs.update(config["dataset"]["arguments"])
    except KeyError:
        pass
    
    datasets = {
        "mnist"       : lambda: vdatasets.load_mnist(**kwargs),
        "olivetti"    : lambda: vdatasets.load_olivetti(**kwargs),
        "blobs"    : lambda: vdatasets.load_blobs(**kwargs),
        "single_cell"    : lambda: vdatasets.load_single_cell(**kwargs),
        "compact"    : lambda: vdatasets.load_compact(**kwargs),
        "scanpy": lambda: vdatasets.load_scanpy(**kwargs),
        "scvelo": lambda: vdatasets.load_scvelo(**kwargs)
    }

    dataset = datasets.get(config["dataset"]["name"].lower(), lambda: None )()
    if dataset is None:
        raise ValueError("Wrong dataset name, available: %s" % datasets.keys())
    return dataset

def get_optimizer(config, sub_model):
    """load an optimizer from a json config file"""
    torch_optimizer = getattr(torch.optim, config["optimizers"][sub_model]["name"])
    optimizer_kwargs = config["optimizers"][sub_model]["args"]
    def _do(params):
        if optimizer_kwargs["lr"] == 0:
            return None
        
        return torch_optimizer(params, **optimizer_kwargs)
    return _do

def load_configuration(jsonfile, get_original=False):
    """load a json confguration file"""
    import json
    import copy
    import urllib.request 

    non_linearities = {
        "sin": torch.sin,
        "relu": torch.nn.ReLU(),
        "leakyrelu": torch.nn.LeakyReLU()
    }

    if "http" in jsonfile or "ftp" in jsonfile:
        with urllib.request.urlopen(jsonfile) as url:
            config = json.loads(url.read().decode())   
    else :
        f = open(jsonfile)
        config = json.load(f)
        f.close()
    
    bck_json = None
    if get_original:
        bck_json = copy.deepcopy(config)
    
    for k, v in config["model"].items():
        if k.find("non_linearity") > -1 :
            config["model"][k] = non_linearities[v.lower()]

    for k, v in config["optimizers"].items():
        config["optimizers"][k] = get_optimizer(config, k)

    if get_original:
        return config, bck_json

    return config

def make_fae_model(config, dataset, model_class, device="cuda", model_filename=None, output_scaling_base=(-1, 1) ):
    from . import persistence as vpers
    from . import nn as vnnutils

    output_transform = None
    if output_scaling_base:
        output_transform = vnnutils.ScaleNonLinearity(-1., 1., dataset["sample_scale"][0], dataset["sample_scale"][1])

    model_args = dict(config["model"])
    model_args.update(
        dict(
            x_dim=dataset["shapes"]["input_size"],
            nb_class=dataset["shapes"]["nb_class"],
            output_transform=output_transform,
        )
    )
    if model_filename:
        model = vpers.load(
            filename=model_filename,
            model_class=model_class,
            map_location=device,
            model_args=model_args
        )
    else :
        model = model_class(**model_args)
        model.to(device)

    return model

def train(model, dataset, config, nb_epochs, run_device=None):
    if run_device is not None:
        config["run_device"] = run_device
    
    trainer = vtrain.Trainer(**config["trainer"])
    
    history = trainer.run(
            model,
            nb_epochs = nb_epochs,
            batch_formater=dataset["batch_formater"],
            train_loader=dataset["loaders"]["train"],
            reconstruction_opt_fct = config["optimizers"]["reconstruction"],
            condition_adv_opt_fct = config["optimizers"]["condition_adv"],
            condition_fit_opt_fct = config["optimizers"]["condition_fit"],
            condition_fit_generator_opt_fct = config["optimizers"]["condition_fit_generator"],
            gan_generator_opt_fct = config["optimizers"]["gan_generator"],
            gan_discriminator_opt_fct = config["optimizers"]["gan_discriminator"],
            train_reconctruction_freq=config["hps"]["train_reconctruction_freq"],
            train_condition_adv_freq=config["hps"]["train_condition_adv_freq"],
            train_gan_discrimator_freq=config["hps"]["train_gan_discrimator_freq"],
            train_gan_generator_freq=config["hps"]["train_gan_generator_freq"],
            train_condition_fit_predictor_freq=config["hps"]["train_condition_fit_predictor_freq"],
            train_condition_fit_generator_freq=config["hps"]["train_condition_fit_generator_freq"],
            projection_l1_compactification=config["hps"].get("projection_l1_compactification", 0.),
            test_loader=None
        )

    return trainer, history