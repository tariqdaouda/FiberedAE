import torch
import numpy

from fiberedae.utils import nn as nnutils


# def init_weights_b(layer):
#     """Init for the bias"""
#     if type(layer) == torch.nn.Linear:
#         torch.nn.init.xavier_uniform_(layer.weight)

# def init_weights(layer, batchsize=256, nonlinearity=torch.sin):
#     """Init for weights"""
#     if type(layer) == torch.nn.Linear:
#         inp = torch.tensor( numpy.random.randn(batchsize, layer.weight.shape[1]), dtype=torch.float )
#         res = nonlinearity( layer(inp) )
#         std = torch.std(res)
#         layer.weight = torch.nn.Parameter(layer.weight / std)

class ConditionHandler(torch.nn.Module):
    """docstring for ConditionHandler"""
    def __init__(self, embedder, fiber_predictor, recons_predictor):
        super(ConditionHandler, self).__init__()
        self.embedder = embedder
        self.fiber_predictor = fiber_predictor
        self.recons_predictor = recons_predictor

    def embed(self, input_x):
        return self.embedder(input_x)

    def predict_from_fiber(self, fiber_x):
        return self.fiber_predictor(fiber_x)
    
    def predict_from_recons(self, recons_x):
        return self.recons_predictor(recons_x)
    
class FlexFAE(torch.nn.Module):
    def __init__(
            self,
            fiber_encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            condition_handlers: dict,
            gan_discriminator: torch.nn.Module
        ):
        super(FiberedAE, self).__init__()
      
        self.fiber_encoder = fiber_encoder
        self.condition_handlers = condition_handlers
        self.decoder = decoder
        self.gan_discriminator = gan_discriminator

    def to(self, device, *args, **kwargs):
        self.run_device = device
        super(FiberedAE, self).to(self.run_device, *args, **kwargs)
        print("running on: %s" % self.run_device)
    
    def cuda(self, *args, **kwargs):
        self.to("cuda", *args, **kwargs)
    
    def _embed_conditions(self, conditions:dict, cat=True):
        res = {
            key: self.condition_handlers[key](cond) for key, value in conditions.items()
        }
        if cat: return torch.cat(res.values())
        return res

    def _condition_adversarial(self, fiber, conditions:dict):
        rev_fiber = grad_reverse(fiber)
        res = {
            key: self.condition_handlers[key].predict_from_fiber(rev_fiber, cond) for key, value in conditions.items()
        }
        return res

    def _condition_fitting(self, recons, conditions:dict):
        res = {
            key: self.condition_handlers[key].predict_from_recons(rev_fiber, cond) for key, value in conditions.items()
        }
        return res

    def forward_all(self, x, conditions:dict):
        fiber = self.fiber_encoder(x)
        cond_adv = self._condition_adversarial(fiber, conditions)
        embs = self._embed_conditions(conditions, cat=False)
        embs.append(fiber)
        embs = torch.cat(embs)

        recons = self.decode(embs)
        gan = self.gan_discriminator(recons)
        cond_fit = self._condition_fitting(recons, conditions)

        return {
            "fiber": fiber,
            "cond_adv": cond_adv,
            "embs": embs,
            "recons": recons,
            "gan": gan,
            "cond_fit": cond_fit,
        }
