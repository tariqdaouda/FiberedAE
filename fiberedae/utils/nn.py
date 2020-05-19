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

class PNonLinearity(torch.nn.Module):
    """Parametrised a*func(x)"""
    def __init__(self, function, init_value=1.):
        super(PNonLinearity, self).__init__()
        self.function = function
        self.init_value = init_value
        self.weights = None

    def forward(self, x):
        if self.weights is None:
            tensor = torch.Tensor(x.size[0]).fill_(self.init_value)
            self.weights = torch.nn.Parameter(tensor)

        val = self.weights * self.function(x)
        return val

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