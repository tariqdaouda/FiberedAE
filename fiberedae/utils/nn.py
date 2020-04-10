import numpy
import torch

##ADAPTED FROM Jean_Da_Rolt: https://discuss.pytorch.org/t/why-cant-i-see-grad-of-an-intermediate-variable/94/21
def get_gradient_printer(str_identifier):
    """This function returns a printer function, that prints information about a  tensor's
    gradient. Used by register_hook in the backward pass.
    """
    def printer(grad_tensor):
        if grad_tensor.nelement() == 1:
            print(f"{str_identifier} {grad_tensor}")
        else:
            print(f"{str_identifier} shape: {grad_tensor.shape}"
                  f" max: {grad_tensor.max()} min: {grad_tensor.min()}"
                  f" mean: {grad_tensor.mean()}")
    return printer

def register_grad_printer_hook(tensor, str_identifier):
    """Utility function to call retain_grad and Pytorch's register_hook
    in a single line
    """
    tensor.retain_grad()
    tensor.register_hook(get_gradient_printer(str_identifier))

class ModuleFunction(torch.nn.Module):
    """Magically transforms any function into pyTorch module
    Useful for using custom non-linearities with pyTocrh Sequencial"""
    def __init__(self, function):
        super(ModuleFunction, self).__init__()
        self.function = function
    
    def forward(self, x):
        return self.function(x)

    def __repr__(self):
        try :
            return "%s(%s)" % (self.__class__.__name__, self.non_linearity.__name__)
        except:
            return "%s(function)" % (self.__class__.__name__)

def get_fc_layer(in_dim, out_dim, non_linearity, batchnorm, bias=True):
    layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
    to_add = [layer]
    
    if batchnorm:
        batchnorm_layer = torch.nn.BatchNorm1d(layer.out_features)
        to_add.append(batchnorm_layer)

    if non_linearity is not None : 
        if not isinstance(non_linearity, torch.nn.Module):
            non_linearity = ModuleFunction(non_linearity)
        to_add.append(non_linearity)

    return torch.nn.Sequential( *to_add )

def get_fc_network(x_dim, h_dim, out_dim, nb_layers, non_linearity, last_non_linearity, h_out_dim=None, batchnorm=False):#, output_batchnorm = False):
    lst = []
    if h_out_dim is None :
        h_out_dim = h_dim
    l_out_dim = h_out_dim
    l_in_dim = x_dim
    current_nl = non_linearity
    for i in range(nb_layers):
        if i > 0 :
            l_in_dim = h_dim
        if i == nb_layers-1:
            current_nl = last_non_linearity
            l_out_dim = out_dim
        layer = get_fc_layer(l_in_dim, l_out_dim, non_linearity=current_nl, batchnorm=batchnorm)
        lst.append(layer)

    seq = torch.nn.Sequential( *lst )
    return seq

def get_gradients_magnitude(model, inputs, targets, criterion) :
    """return the absolute sum of gradients"""
    optimizer = torch.optim.SGD
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = criterion(outputs, targets)
    return numpy.sum(numpy.abs(list(network.parameters())))

def get_loss(model, inputs, targets, criterion) :
    """return the loss"""
    outputs = model(**inputs)
    loss = criterion(outputs, targets)
    return loss.item()

class ScaleNonLinearity(torch.nn.Module):
    """ScaleNonLinearity a non linearity to be in the correct range"""
    def __init__(self, original_min, original_max, final_min, final_max):
        super(ScaleNonLinearity, self).__init__()
        self.original_min = original_min
        self.original_max = original_max
        self.final_min = final_min
        self.final_max = final_max
    
    def forward(self, value):
        original_scale = self.original_max - self.original_min
        final_scale = self.final_max - self.final_min
        
        value = ( value - self.original_min) * ( final_scale / original_scale ) + self.final_min
        return value


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
    AVAILABLE_PROJECTION_TYPES = ["torus", "cube"]
    
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
            raise ValueError("Unknown projection type: %s, available: %s" % (projection_type, self.AVAILABLE_PROJECTION_TYPES) )
        return proj

    def forward(self, x):
        out = self.pre_projection(x)
        out = self.project(out)
        return out