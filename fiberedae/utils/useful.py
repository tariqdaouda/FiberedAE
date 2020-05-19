import torch
import fiberedae.utils.datasets as vdatasets
import fiberedae.utils.basic_trainer as vtrain

def load_dataset(config):
    datasets = {
        "mnist"       : lambda: vdatasets.load_mnist(config["hps"]["minibatch_size"]),
        "olivetti"    : lambda: vdatasets.load_olivetti(config["hps"]["minibatch_size"]),
        "blobs"    : lambda: vdatasets.load_blobs(config["hps"]["minibatch_size"]),
        "single_cell"    : lambda: vdatasets.load_single_cell(
            batch_size=config["hps"]["minibatch_size"],
            condition_field=config["dataset"]["condition_field"],
            filepath=config["dataset"]["filepath"],
            dataset_name=config["dataset"]["dataset_name"],
            backup_url=config["dataset"]["backup_url"]
        ),
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

def load_configuration(jsonfile):
    """load a json confguration file"""
    import json
    non_linearities = {
        "sin": torch.sin,
        "relu": torch.nn.ReLU(),
        "leakyrelu": torch.nn.LeakyReLU()
    }

    with open(jsonfile) as f:
        config = json.load(f)
        for k, v in config["model"].items():
            if k.find("non_linearity") > -1 :
                config["model"][k] = non_linearities[v.lower()]

        for k, v in config["optimizers"].items():
            config["optimizers"][k] = get_optimizer(config, k)

    return config

def train(model, dataset, config, nb_epochs):
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
            test_loader=None
        )

    return trainer, history