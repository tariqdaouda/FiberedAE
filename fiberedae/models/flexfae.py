import torch
import numpy

from fiberedae.utils import nn as nnutils


AVAILABLE_PROJECTION_TYPES = ["torus", "cube"]

def init_weights_b(layer):
    """Init for the bias"""
    if type(layer) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)

def init_weights(layer, batchsize=256, nonlinearity=torch.sin):
    """Init for weights"""
    if type(layer) == torch.nn.Linear:
        inp = torch.tensor( numpy.random.randn(batchsize, layer.weight.shape[1]), dtype=torch.float )
        res = nonlinearity( layer(inp) )
        std = torch.std(res)
        layer.weight = torch.nn.Parameter(layer.weight / std)

class GradientReverse(torch.autograd.Function):
    """Implementation of gradient reverse layer"""
    scale = 1.0
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        new_grads = GradientReverse.scale * grad_output.neg()
        return new_grads
    
def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)

class InputResidualSequential(torch.nn.Module):
    """Propagates through the network while concatenating a skip input at every level"""
    def __init__(self, layers):
        super(InputResidualSequential, self).__init__()
        self.layers = layers

    def forward(self, x, skip_input, layer_id=None) :
        if layer_id is not None:
            assert abs(layer_id) <= len(self) -1

        if layer_id is not None and layer_id < 0:
            layer_id = len(self) + layer_id

        layer_in = x
        out = None
        for i, layer in enumerate(self.layers) :
            out = layer(layer_in) 
            layer_in = torch.cat( [out, skip_input], 1 )
            
            if layer_id == i:
                break
        return out

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)

class MLPClassifier(torch.nn.Module):
    """docstring for MLPClassifier"""
    def __init__(self, x_dim, nb_classes, nb_layers, h_dim, non_linearity, sigmoid_out=False):
        super(MLPClassifier, self).__init__()
        assert nb_classes > 1

        if sigmoid_out :
            last_nl = torch.nn.Sigmoid()
            out_dim = 1
        else :
            last_nl = torch.nn.Softmax(dim=1)
            out_dim = nb_classes

        layers = nnutils.get_fc_network(
            x_dim=x_dim,
            h_dim=h_dim,
            out_dim=out_dim,
            nb_layers=nb_layers,
            non_linearity=non_linearity,
            last_non_linearity=last_nl
        )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x) :
        return self.layers(x)

class FiberSpace(torch.nn.Module):
    def __init__(self, x_dim, h_dim, fiber_dim, nb_layers, non_linearity, projection_type, projection_batchnorm):
        super(FiberSpace, self).__init__()
        if projection_type == "torus" and fiber_dim % 2 != 0 :
            raise ValueError("for torus projection fiber_dim must be a multiple of 2")
        
        self.pre_projection = nnutils.get_fc_network(
                x_dim=x_dim,
                h_dim=h_dim,
                out_dim=h_dim,
                nb_layers=nb_layers,
                non_linearity=non_linearity,
                last_non_linearity=non_linearity
            )

        self.projection = nnutils.get_fc_layer(
            in_dim=h_dim,
            out_dim=fiber_dim,
            non_linearity=None,
            batchnorm = projection_batchnorm
        )
        self.out_dim = fiber_dim
        self.projection_type = projection_type        
        self.projection.apply(init_weights)

    def project(self, x):
        proj = self.projection(x)
        if self.projection_type == "cube":
            proj = torch.sin(proj)
        elif self.projection_type == "torus":
            half = self.out_dim//2
            sin, cos = torch.sin(proj[:, :half]), torch.cos(proj[:, half:])
            proj = torch.cat([sin, cos], 1)
        else :
            raise ValueError("Unknown projection type: %s, available: %s" % (projection_type, AVAILABLE_PROJECTION_TYPES) )
        return proj

    def forward(self, x):
        out = self.pre_projection(x)
        out = self.project(out)
        return out

# class FiberHandler(torch.nn.Module):
#     """docstring for FiberHandler"""
#     def __init__(self, *args, **kwargs):
#         super(FiberHandler, self).__init__()
    
#     def embed(self, input_x):
#         raise NotImplemented("Abstract function")
    
# class DecoderHandler(torch.nn.Module):
#     """docstring for DecoderHandler"""
#     def __init__(self, *args, **kwargs):
#         super(DecoderHandler, self).__init__()
    
#     def decode(self, input_x):
#         raise NotImplemented("Abstract function")
    
# class FiberedAE(torch.nn.Module):
#     """docstring for FiberedAE"""
#     def __init__(self, condition_hand, fiber_hand, decoder_hand):
#         super(FiberedAE, self).__init__()
#         self.condition_hand = condition_hand
#         self.fiber_hand = fiber_hand
#         self.decoder_hand = decoder_hand

#     def forward(self, input_x):
#         raise NotImplemented("Abstract function")

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
