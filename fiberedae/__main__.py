import fiberedae.models.fae as vmod
import fiberedae.utils.basic_trainer as vtrain
import fiberedae.utils.persistence as vpers
import fiberedae.utils.datasets as vdatasets
import fiberedae.utils.useful as us

# import argparse
import random
import os
import json
import click

def get_folder_name(folder, overwrite):
    import time
    if overwrite:
        return folder
    suffix = time.ctime().replace(" ", "-")
    return "%s-%s" %(folder, suffix) 


def get_quote():
    epictetus = [
        "It's not what happens to you, but how you react to it that matters.",
        "We have two ears and one mouth so that we can listen twice as much as we speak.",
        "Only the educated are free.",
        "He who laughs at himself never runs out of things to laugh at.",
        "It is impossible for a man to learn what he thinks he already knows."
    ]

    quote = "%s -- Epictetus" % random.choice(epictetus)
    sep = "="* (len(quote) + 4)
    ret = []
    ret.append(sep)
    ret.append("| " + quote + " |")
    ret.append(sep)
    return "\n".join(ret)

@click.group()
def main():
    click.echo(get_quote())

@main.command()
@click.argument("configuration_file")
@click.option("-sci", "--sc_input_file", help="Override the single cell dataset .h5 defined in the json. Use with care")
@click.option("-scc", "--sc_condition", help="Override the condition for a single cell dataset. Use with care")
@click.option("-scb", "--sc_backup", help="Override the backup url for a single cell dataset. Use with care")
@click.option("-n", "--experiment_name", help="experiment name", default=None)
@click.option("-e", "--epochs", help="bypass epochs value in configuration", type=int, default=-1)
@click.option("-m", "--model", help="load a previously trained model", default=None)
@click.option("--device", help="cpu, cuda, ...", type=str, default="cuda")
@click.option("--overwrite/--no_overwrite", default=False)
def train(**args):
    import torch
    import torch.nn as nn

    print(args)
    if args["experiment_name"] :
        print("\t creating folder...")
        exp_folder = get_folder_name(args["experiment_name"], args["overwrite"])
        try:
            os.mkdir(exp_folder)
        except FileExistsError:
            pass
    else :
        exp_folder = "."

    print("\t loading configuration...")
    config, orig_conf = us.load_configuration(args["configuration_file"], get_original=True)
    print("args:", args)
    for arg_key, json_key in [("sc_input_file", "filepath"), ("sc_condition", "condition_field"), ("sc_backup", "backup_url")]:
        if args[arg_key]:
            print("\t-> Overriding", json_key, "with:", args[arg_key])
            config["dataset"]["arguments"][json_key] = args[arg_key]
            orig_conf["dataset"]["arguments"][json_key] = args[arg_key]

    print("\t loading dataset...")
    dataset = us.load_dataset(config)

    print("\t making model...")

    model = us.make_fae_model(
        config=config,
        dataset=dataset,
        model_class=vmod.FiberedAE,
        device = args["device"],
        model_filename=args["model"]
    )

    print("---" )
    print("Available GPUs: ", torch.cuda.device_count() )
    print("---" )
    if torch.cuda.device_count() > 1:
        print("\t\t Launching in || mode")
        model = nn.DataParallel(model)
    
    print("\t training...")
    if args["epochs"] > 0:
        orig_conf["hps"]["nb_epochs"] = args["epochs"]
    
    trainer, history = us.train(model, dataset, config, nb_epochs=orig_conf["hps"]["nb_epochs"], run_device=args["device"])

    print("\t saving model...")
    vpers.save(
        model,
        filename=os.path.join(exp_folder, "model.pt"),
        training_history=history,
        meta_data=None,
        condition_encoding=dataset["label_encoding"],
        model_args=None, # buggy pytorch pkl save
   )

    print("\t saving config...")
    with open(os.path.join(exp_folder, "configuration.json"), "w") as f:
        json.dump(orig_conf, f, indent=4)

@main.command()
@click.argument("configuration_file")
@click.argument("model", default=None)
@click.option("-r", "--reference", help="the reference condition to which translation should be made. The default is whichever one comes is first")
@click.option("-d", "--dataset", help="override dataset in config file")
@click.option("-b", "--batch_size", help="size of the minibatch", default=1024)
@click.option("-o", "--output", help="the final h5ad name")
@click.option("--device", help="cpu, cuda, ...", default="cpu")
@click.option("--print_references", help="print available references and exit")
def translate_single_cell(**args):
    """Translate a single cell dataset into a reference condition"""
    import fiberedae.utils.single_cell as vsc

    print("\t loading configuration...")
    config, orig_conf = us.load_configuration(args["configuration_file"], get_original=True)
    if args["dataset"]:
        print("\t replacing dataset:\n\t\t%s\n\t\tby:\n\t\t%s" % (config["dataset"]["arguments"]["filepath"], args["dataset"]))
        config["dataset"]["arguments"]["filepath"] = args["dataset"] 
        orig_conf["dataset"]["arguments"]["filepath"] = args["dataset"] 
    
    print("\t loading dataset...")
    dataset = us.load_dataset(config)
    condition_key = config["dataset"]["arguments"]["condition_field"]

    if args["reference"] is None :
        args["reference"] = dataset["adata"].obs[condition_key].unique()[0]

    if args["print_references"]:
        print("\tAvailable refrences:")
        try:
            print("\t\t", dataset["adata"].obs[condition_key].unique())
        except Exception as e:
            try:
                print("\t\t", dataset["adata"].obsm[condition_key].unique())
            except Exception as e:
                raise KeyError("nor obs nor obsm contain field:", condition_key)
        return

    print("loading model...")
    model = us.make_fae_model(
        config=config,
        dataset=dataset,
        model_class=vmod.FiberedAE,
        device = args["device"],
        model_filename=args["model"]
    )

    print("translating...")
    res = vsc.translate(
        model = model,
        adata = dataset["adata"],
        condition_key = condition_key,
        ref_condition = args["reference"],
        condition_encoder = dataset["label_encoding"],
        batch_size=args["batch_size"]
    )

    # print("adding X_fae...")
    # ret = vsc.reconstruct(
    #     model = model,
    #     adata = dataset["adata"],
    #     run_device = args["device"],
    #     batch_size=args["batch_size"],
    #     cleaned_output = False,
    #     fiber_output = True
    # )
    # res.obsm["X_fae"] = ret["X_fiber"]

    name = config["dataset"]["arguments"]["dataset_name"].replace(" ", "-")
    if not args["output"]:
        args["output"] =  name + "-ref_" + args["reference"] + ".h5ad"

    print("saving result to: %s..." % args["output"])
    res.write(args["output"])

@main.command()
@click.argument("configuration_file")
@click.argument("model", default=None)
@click.option("-b", "--batch_size", help="size of the minibatch", default=128)
@click.option("-o", "--output", help="the final h5ad name")
@click.option("--device", help="cpu, cuda, ...", default="cpu")
def clean_single_cell(model, adata, run_device):
    """
    Reconstruct the input and cleans it
    if fiber_output returns fiber layer embeddings instead of reconstruction
    """
    import fiberedae.utils.single_cell as vsc

    print("\t loading configuration...")
    config, orig_conf = us.load_configuration(args["configuration_file"], get_original=True)
    if args["dataset"]:
        print("\t replacing dataset:\n\t\t%s\n\t\tby:\n\t\t%s" % (config["dataset"]["arguments"]["filepath"], args["dataset"]))
        config["dataset"]["arguments"]["filepath"] = args["dataset"] 
        orig_conf["dataset"]["arguments"]["filepath"] = args["dataset"] 

    print("\t loading dataset...")
    dataset = us.load_dataset(config)
    condition_key = config["dataset"]["arguments"]["condition"]

    print("loading model...")
    model = us.make_fae_model(
        config=config,
        dataset=dataset,
        model_class=vmod.FiberedAE,
        device = args["device"],
        model_filename=args["model"]
    )

    print("cleaning...")
    ret = vsc.reconstruct(
        model = model,
        adata = dataset["adata"],
        run_device = device,
        batch_size=args["batch_size"],
        clean_output = True,
        fiber_output = True
    )

    adata.X = ret["X"]
    adata.obsm["X_fae"] = ret["X_fiber"]

    name = config["dataset"]["arguments"]["dataset_name"].replace(" ", "-")
    if not args["output"]:
        args["output"] =  name + "-fae-cleaned.h5ad"

    print("saving result to: %s..." % args["output"])
    adata.write(args["output"])

if __name__ == "__main__" :
    main()
