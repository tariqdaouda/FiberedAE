import torch

def save_model(model, optimizers, training_history, meta_data, condition_encoding, model_creation_args, filename):
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
    # map_location='cpu' is necessary to load model trained on cuda for a cpu only machine
    state = torch.load(filename, map_location=map_location)
    model = model_class(**state['model_creation_args'])
    model.load_state_dict(state['model_state_dict'])

    optimizers = {}
    for name, data in state["optimizers"].items():
        optimizer = getattr(torch.optim, data['class_name'] )( model.parameters())
        optimizers[name] = optimizer
    
    return model, optimizers, state["training_history"], state['condition_encoding']