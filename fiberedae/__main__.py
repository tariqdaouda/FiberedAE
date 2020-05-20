import fiberedae.models.fae as vmod
import fiberedae.utils.basic_trainer as vtrain
import fiberedae.utils.persistence as vpers
# import fiberedae.utils.single_cell as vsc
import fiberedae.utils.datasets as vdatasets
import fiberedae.utils.useful as us
import fiberedae.utils.nn as vnnutils

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
    parser.add_argument("-m", "--model", help="load a previously trained model", type=str, action="store", default="")
    parser.add_argument("--device", help="cpu, cuda, ...", type=str, default="cuda")
    parser.add_argument("--no_overwrite", help="If true will create an ew folder each time", type=bool, default=True, action="store")
    
    args=parser.parse_args().__dict__


    print("\t creating folder...")
    exp_folder = get_folder_name(args["experiment_name"], not args["no_overwrite"])
    os.mkdir(exp_folder)

    print("\t loading configuration...")
    config, orig_conf = us.load_configuration(args["configuration_file"], get_original=True)
    print(orig_conf)

    print("\t loading dataset...")
    dataset = us.load_dataset(config)

    print("\t making model...")
    #make model
    model_args = dict(config["model"])
    model_args.update(
        dict(
            x_dim=dataset["shapes"]["input_size"],
            nb_class=dataset["shapes"]["nb_class"],
            output_transform=vnnutils.ScaleNonLinearity(-1., 1., dataset["sample_scale"][0], dataset["sample_scale"][1]),
        )
    )
    if args["model"] != "" :
        model = vpers.load(
            filename=args["model"],
            model_class=vmod.FiberedAE,
            map_location=args["device"],
            model_args=model_args
        )
    else :
        model = vmod.FiberedAE(**model_args)
        model.to(args["device"])

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
        json.dump(orig_conf, f)


if __name__ == "__main__" :
    main()
