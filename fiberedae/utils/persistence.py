import torch
import pickle
import json

def save_model(model, optimizers, training_history, meta_data, condition_encoding, model_creation_args, filename):
    """No longer works with the last version of pytorch"""
    optimizers_states = {}
    for name, opt in optimizers.items():
        if opt is None :
            optimizers_states[name] = None 
        else :
            optimizers_states[name] = {
                "state_dict": opt.state_dict(),
                "class_name": opt.__class__.__name__
            }

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizers': optimizers_states,
        'training_history': training_history,
        'meta': meta_data,
        'condition_encoding': condition_encoding,
        'model_creation_args': model_creation_args
    }, filename)

def load_model(filename, model_class, map_location):
    """No longer works with the last version of pytorch"""

    # map_location='cpu' is necessary to load model trained on cuda for a cpu only machine
    state = torch.load(filename, map_location=map_location)
    model = model_class(**state['model_creation_args'])
    model.load_state_dict(state['model_state_dict'])

    optimizers = {}
    for name, data in state["optimizers"].items():
        optimizer = getattr(torch.optim, data['class_name'] )( model.parameters())
        optimizers[name] = optimizer
    
    return model, optimizers, state["training_history"], state['condition_encoding']

def serialize(obj, filename):
    if filename.endswith(".json"):
        with open(filename, "w") as fp:
            json.dump(obj, fp)
    elif filename.endswith(".pt"):
        torch.save(obj.state_dict(), filename)
    else:
        with open(filename, "wb") as fp:
            pickle.dump(obj, fp)

def deserialize(filename, map_location="cuda"):
    if filename.endswith(".json"):
        with open(filename, "r") as fp:
            obj = json.load(fp)
    elif filename.endswith(".pt"):
        obj = torch.load(filename , map_location=map_location)
    else:
        with open(filename, "rb") as fp:
            obj = pickle.load(fp)
    return obj

def save(
        model,
        filename,
        # optimizers=None,
        training_history=None,
        meta_data=None,
        condition_encoding=None,
        model_args=None,
    ):

    model_fn = str(filename)
    if not filename.endswith(".pt"):
        model_fn = filename + ".pt"

    serialize(model, model_fn)

    if training_history:
        serialize(training_history, filename[:-3] + "_curves.pkl")

    if meta_data:
        serialize(meta_data, filename[:-3] + "_metadata.pkl")
    
    if condition_encoding:
        serialize(condition_encoding, filename[:-3] + "_encoding.pkl")
    
    if model_args:
        serialize(model_args, filename[:-3] + "_args.pkl")

def load(filename, model_class, map_location, model_args=None):
    # if not filename.endswith(".pt"):
        # raise ValueError("Filename must end with .pt")

    state = deserialize(filename, map_location)
    if not model_args:
        try:
            args = deserialize(filename.replace(".pt", "_args.pkl"))
        except Exception as e:
            print("Unable to load model arguments, please provide them as function arguments")
            raise e
    else :
        args = model_args
    
    model = model_class(**args)
    model.load_state_dict(state)

    return model

def load_folder(folder_path, model_class, map_location, model_args=None):
    res = {}
    model_filename = folder_path + "model.pt"
    res["model"] = load(model_filename, model_class, map_location, model_args)
    for data_type in ["metadata", "curves", 'encoding', "args"]:
        filename = model_filename[:-3] + "_%s.pkl" % data_type
        try :
            res[data_type] = deserialize(filename)
        except Exception as e:
            print ("Unable to load %s because of %s" % (filename, e))
            res[data_type] = None

    return res
