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
        assert nb_classes >= 1

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
        
        tensor = torch.Tensor(fiber_dim).fill_(1.)
        self.weights = torch.nn.Parameter(tensor, dtype=torch.float)

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
    
        proj = self.weights * proj
        return proj

    def forward(self, x):
        out = self.pre_projection(x)
        out = self.project(out)
        return out

class FiberedAE(torch.nn.Module):
    def __init__(self,
        x_dim,
        h_dim,
        base_dim,
        fiber_dim,
        nb_class,
        non_linearity,
        nb_hidden,
        nb_fiber_hidden,
        condition_adv_h_dim,
        condition_adv_nb_hidden,
        condition_adv_non_linearity,
        condition_fit_h_dim,
        condition_fit_nb_hidden,
        condition_fit_non_linearity,
        gan_discriminator_h_dim,
        gan_discriminator_nb_hidden,
        gan_discriminator_non_linearity,
        fiber_batchnorm,
        conditioned,
        residual_fiber,
        residual_base,
        contract_fiber,
        contract_base,
        projection_type,
        pnon_linearity_l1=0.,
        wgan=False,
        output_transform=None,
    ):
        super(FiberedAE, self).__init__()
      
        if pnon_linearity_l1 > 0:
            non_linearity = nnutils.PNonLinearity(non_linearity)

        self.fiber = FiberSpace(    # Encoder
            x_dim=x_dim,             # Xdim (Math renaming scheme)
            h_dim=h_dim,             # Ydim (Math renaming scheme)
            fiber_dim=fiber_dim,             # Fdim (Math renaming scheme)
            nb_layers=nb_fiber_hidden,
            non_linearity=non_linearity,
            projection_type=projection_type,
            projection_batchnorm=fiber_batchnorm
        )
        self.fiber.apply(init_weights)

        backbone_in_h_dim = h_dim       # Ydim
        if residual_fiber:
            backbone_in_h_dim += self.fiber.out_dim
        if residual_base:
            backbone_in_h_dim += base_dim  # Bdim

        self.backbone = nnutils.get_fc_network(
            x_dim=base_dim + self.fiber.out_dim,
            h_dim=backbone_in_h_dim,
            h_out_dim=h_dim,
            out_dim=x_dim,
            nb_layers=nb_hidden,
            non_linearity=non_linearity,
            last_non_linearity=non_linearity,
        )

        if residual_base or residual_fiber:
            self.backbone = InputResidualSequential(self.backbone)
        self.backbone.apply(init_weights)
        
        self.output_transform = output_transform
        if self.output_transform is not None :
            self.output_transform = nnutils.ModuleFunction(self.output_transform)

        self.conditions = torch.nn.Embedding(nb_class, base_dim)
        self.fiber_condition_adv = MLPClassifier(
            self.fiber.out_dim,
            nb_class,
            nb_layers=condition_adv_nb_hidden,
            h_dim=condition_adv_h_dim,
            non_linearity=condition_adv_non_linearity
        )
        self.fiber_condition_adv.apply(init_weights)
        
        self.output_classifier = MLPClassifier(
            x_dim,
            nb_class,
            nb_layers=condition_fit_nb_hidden,
            h_dim=condition_fit_h_dim,
            non_linearity=condition_fit_non_linearity
        )
        self.output_classifier.apply(init_weights)
        
        if wgan:
            output_gan_discriminator = nnutils.get_fc_network(
                x_dim=x_dim,
                h_dim=gan_discriminator_h_dim,
                out_dim=1,
                nb_layers=gan_discriminator_nb_hidden,
                non_linearity=gan_discriminator_non_linearity,
                last_non_linearity=None
            )
            self.output_gan_discriminator = torch.nn.Sequential(*output_gan_discriminator)
        else :
            self.output_gan_discriminator = MLPClassifier(
                x_dim,
                nb_classes=2,
                nb_layers=gan_discriminator_nb_hidden,
                h_dim=gan_discriminator_h_dim,
                non_linearity=gan_discriminator_non_linearity,
                sigmoid_out=True
            )
        self.output_gan_discriminator.apply(init_weights)

        self.pnon_linearity_l1 = pnon_linearity_l1
        self.wgan = wgan
        self.last_in = None
        self.last_out = None
        self.last_fiber = None
        self.gan_targets = None
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.nb_hidden = nb_hidden
        self.nb_fiber_hidden = nb_fiber_hidden
        self.base_dim = base_dim
        self.fiber_dim = fiber_dim
        self.nb_class= nb_class

        self.non_linearity = non_linearity
        self.run_device = "cpu"

        self.fiber_grads = None
        self.conditioned = conditioned
        self.residual_fiber = residual_fiber
        self.residual_base = residual_base

        self.contract_fiber = contract_fiber
        self.contract_base = contract_base
        self.contractables_grads = {}

    def _save_grads_hook(self, tensor, str_id):
        """store the intermidate jacobian for tensor"""
        def _hook(grad):
            self.contractables_grads[str_id] = grad
        tensor.requires_grad_(True)
        tensor.retain_grad()
        tensor.register_hook(_hook)

    def to(self, device, *args, **kwargs):
        self.run_device = device
        super(FiberedAE, self).to(self.run_device, *args, **kwargs)
        print("running on: %s" % self.run_device)
    
    def cuda(self, *args, **kwargs):
        self.to("cuda", *args, **kwargs)
    
    def condition(self, cond):
        # Bug fix: https://github.com/ictnlp-wshugen/annotated-transformer_codes/commit/ffe3bcc2665fbe5a7f1d53ca8819b1a455903cb8
        # For windows, condition has to be of type LongTensor and not Int
        # condition = self.conditions(cond).view((-1, self.base_dim))
        condition = self.conditions( cond.long() ).view((-1, self.base_dim))
        return condition
            
    def forward_output(self, x, cond, fiber_input=False):
        """
        Additional parameters for quick and dirty prototyping:

        fiber_input       : Misleading.
                          If false, x is fed to fibererver/encoder
                          If true , x is already a would output of fibererver/encoder. Why detach by default? Speed up?
        """
        if not fiber_input:
            fiber = self.fiber(x)
        else:
            fiber = x

        condition = self.condition(cond)

        if not self.conditioned:
            condition -= condition
        
        if self.training:
            if self.contract_fiber:
                self._save_grads_hook(fiber, "fiber_value")
            
            if self.contract_base:
                self._save_grads_hook(condition, "condition")
        
        out = torch.cat( (fiber, condition), 1)
        if self.residual_fiber and self.residual_base:
            out = self.backbone(out, skip_input=out)
        elif self.residual_fiber:
            out = self.backbone(out, skip_input=fiber)
        elif self.residual_base:
            out = self.backbone(out, skip_input=condition)
        else:
            out = self.backbone(out)

        if self.output_transform is not None:
            out = self.output_transform(out)
        
        if not fiber_input:
            self.last_in = x
    
        self.last_out = out
        self.last_fiber = fiber

        return out

    def forward_decode(self, encoding):
        """
        encoding: a concatenantion of fiber and condition [fiber, condition]. Does not support residual fiber or cond alone.
        """

        if self.residual_fiber and self.residual_base:
            out = self.backbone(encoding, skip_input=encoding)
        else:
            out = self.backbone(encoding)

        if self.output_transform is not None:
            out = self.output_transform(out)
        
        return out

    def predict_fiber_condition(self, fiber=None):
        if self.fiber_condition_adv is None:
            return None
        
        if fiber is None :
            fiber = self.last_fiber
        
        fiber = grad_reverse(fiber)
        prediction = self.fiber_condition_adv(fiber)
        return prediction

    def predict_condition(self, last_out=None):
        if self.output_classifier is None:
            return None
        
        if last_out is None :
            last_out = self.last_out#.detach()
        
        prediction = self.output_classifier(last_out)
        return prediction

    def predict_gan(self, last_out=None):
        if self.output_gan_discriminator is None:
            return None
        
        if last_out is None :
            last_out = self.last_out
        
        out = self.output_gan_discriminator(last_out)
        return out

    def forward(self, x, cond):
        return (
            self.forward_output(x, cond),
            self.predict_fiber_condition(),
            self.predict_condition(),
            self.predict_gan()
        )
