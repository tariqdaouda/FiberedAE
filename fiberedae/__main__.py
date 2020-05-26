import fiberedae.models.fae as vmod
import fiberedae.utils.basic_trainer as vtrain
import fiberedae.utils.persistence as vpers
# import fiberedae.utils.single_cell as vsc
import fiberedae.utils.datasets as vdatasets
import fiberedae.utils.useful as us

import argparse
import random
import os
import json


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

print(get_quote())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("configuration_file", help="load the configuration file", type=str, action="store")
    parser.add_argument("experiment_name", help="experiment name", type=str, action="store")
    parser.add_argument("-e", "--epochs", help="bypass epochs value in configuration", type=int, default=-1, action="store")
    parser.add_argument("-m", "--model", help="load a previously trained model", type=str, action="store", default=None)
    parser.add_argument("--device", help="cpu, cuda, ...", type=str, default="cuda")
    parser.add_argument("--no_overwrite", help="If true will create an new folder each time", action="store_false")
    
    args=parser.parse_args().__dict__


    print("\t creating folder...")
    exp_folder = get_folder_name(args["experiment_name"], args["no_overwrite"])
    try:
        os.mkdir(exp_folder)
    except FileExistsError:
        pass

    print("\t loading configuration...")
    config, orig_conf = us.load_configuration(args["configuration_file"], get_original=True)
    
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

    print("\t training...")
    if args["epochs"] > 0:
        orig_conf["hps"]["nb_epochs"] = args["epochs"]
    
    trainer, history = us.train(model, dataset, config, nb_epochs=orig_conf["hps"]["nb_epochs"])

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

def translate_single_cell():
    """Translate a single cell dataset into a reference condition"""
    parser=argparse.ArgumentParser()
    parser.add_argument("configuration_file", help="load the configuration file", type=str, action="store")
    parser.add_argument("reference", help="the reference condition to which translation should be made", type=str, action="store")
    parser.add_argument("model", help="load a previously trained model", type=str, action="store", default=None)
    parser.add_argument("-b", "--batch_size", help="size of the minibatch", type=int, action="store", default=1024)
    parser.add_argument("-o", "--output", help="the final h5ad name", type=str, default=None, action="store")
    parser.add_argument("--device", help="cpu, cuda, ...", type=str, default="cpu")
    
    args=parser.parse_args().__dict__

    print("\t loading configuration...")
    config, orig_conf = us.load_configuration(args["configuration_file"], get_original=True)
    
    print("\t loading dataset...")
    dataset = us.load_dataset(config)

    print("loading model...")
    loaded = vpers.load_folder(filename, model_class, map_location, model_args=None):

    print("translating...")
    
    res = vsc.translate(
        model = loaded["model"],
        adata = dataset,
        condition_key = config["arguments"]["condition_field"],
        ref_condition = args["reference"],
        condition_encoder = loaded["encoding"],
        batch_size=args["batch_size"]
    )

    if not args["output"]:
        args["output"] = args["model"] + "-transref_" + args["reference"] + ".h5ad"

    print("saving result to: %s..." % args["output"])
    res.write(args["output"])

if __name__ == "__main__" :
    main()
