import numpy
import torch

def get_latent_space(model, data_loader, label_encoding, batch_formater, backbone_layer_ids=None, reference_condition=None, run_device="cpu") :
    """return latent spaces organised in a dict"""
    model.eval()
    if reference_condition:
        ref_cond = label_encoding.transform([reference_condition])[0]

    fiber_space = []
    condition = []
    labels = []
    backbone_outputs = None
    if backbone_layer_ids :
        backbone_outputs = { layer_id: [] for layer_id in backbone_layer_ids} 

    for batch in data_loader:
        data, target = batch_formater(batch)
        labels.append(target)
        if reference_condition :
            target = torch.tensor(numpy.zeros(len(data), dtype="int") + ref_cond)
        
        data = data.to(run_device)
        target = target.to(run_device)

        obs = model.fiber(data)
        
        cond = model.condition(target)
        if backbone_outputs:
            for layer_id in backbone_outputs:
                value = model.backbone(condition = cond, obs_value = obs, layer_id = layer_id)
                value = value.cpu().detach().numpy()
                backbone_outputs[layer_id].append(value)

        obs = obs.cpu().detach().numpy()
        cond = cond.cpu().detach().numpy()
        fiber_space.append(obs)
        condition.append(cond)

    labels = numpy.concatenate(labels)
    fiber_space = numpy.concatenate( fiber_space )
    condition = numpy.concatenate( condition )
    if backbone_outputs:
        for layer_id in backbone_outputs:
            backbone_outputs[layer_id] = numpy.concatenate( backbone_outputs[layer_id] )

    return {
        "fiber_space": fiber_space,
        "base_space": condition,
        "labels": labels,
        "str_labels": label_encoding.inverse_transform(labels),
        "backbone_outputs": backbone_outputs 
    }
  
def translate(model, ref_condition, dataloader, batch_formater, run_device="cpu"):
    """Naive transfer implementation using the network only (No geodesic transport)"""
    outs = []
    conds = []
    for batch in dataloader:
        samples, condition = batch_formater(batch)            
        samples = samples.to(model.run_device)
        if ref_condition is not None:
            cond = condition - condition + ref_condition
        else:
            cond = condition

        cond = cond.to(run_device)
        
        out = model.forward_output(x=samples, cond=cond)
        out = out.detach().cpu().numpy()
        cond = cond.detach().cpu().numpy()
        outs.append(out)
        conds.append(condition)
    
    return numpy.concatenate(outs), numpy.concatenate(conds)
