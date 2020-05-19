import fiberedae.models.fae as vmod
import fiberedae.utils.basic_trainer as vtrain
import fiberedae.utils.persistence as vpers
import fiberedae.utils.single_cell as vsc
import fiberedae.utils.datasets as vdatasets
import fiberedae.utils.useful as us

import argparse
import random
import os

epictetus = [
    "It's not what happens to you, but how you react to it that matters.",
    "We have two ears and one mouth so that we can listen twice as much as we speak.",
    "Only the educated are free.",
    "He who laughs at himself never runs out of things to laugh at.",
    "It is impossible for a man to learn what he thinks he already knows."
]

def get_folder_name(folder):
    import time
    suffix = time.ctime().replace(" ", "-")
    return "%s-%s" %(folder, suffix) 

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("configuration_file", help="load the configuration file", type=str, action="store")
    parser.add_argument("experiment_name", help="experiment name", type=str, action="store")
    parser.add_argument("-e", "--epochs", help="bypass epochs value in configuration", type=int, default=-1, action="store")
    parser.add_argument("-m", "--model", help="load a previously trained model", type=str, action="store", default="")
    parser.add_argument("--device", help="cpu, cuda, ...", type=str, default="gpu")
    
    args=parser.parse_args().__dict__

    print("%s -- Epictetus" % random.choice(epictetus))
    
    print("creating folder...")
    exp_folder = get_folder_name(args["name"])
    os.mkdir(exp_folder)


    config = us.load_configuration(args["configuration_file"])

    if args["model"] != "" :
        model = vpers.load_model(args["model"], vmod.FiberedAE, args["device"])[0]
    else :
        #make model
        model_args = dict(config["model"])
        model_args.update(
            dict(
                x_dim=dataset["shapes"]["input_size"],
                nb_class=dataset["shapes"]["nb_class"],
                output_transform=vnnutils.ScaleNonLinearity(-1., 1., dataset["sample_scale"][0], dataset["sample_scale"][1]),
            )
        )
        model = vmod.FiberedAE(**model_args)
        model.to(args["device"])

    print("training...")
    if args["epochs"] > 0:
        config["hps"]["nb_epochs"] = args["epochs"]
    
    trainer, history = us.train(model, dataset, config, nb_epochs=epochs)

    print("saving model...")
    vpers.save_model(
        model, {}, history, {}, dataset["label_encoding"], model_args,
        os.path.join(exp_folder, "model.pytorch.mdl")
    )

    print("saving config...")
    with open(os.path.join(exp_folder, "configuration.json")) as f:
        json.dump(config)


if __name__ == "__main__" :
    main()