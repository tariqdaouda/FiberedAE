from tqdm                  import tqdm
from hnapub.utils.wavelets import schauder_basis
import torch
import numpy as np


class NumericalGeodesics():
    """docstring for Numerical"""
    def __init__(self, n_max, step_count):
        super(NumericalGeodesics, self).__init__()

        self.n_max      = n_max
        self.step_count = step_count
        self.time_grid  = torch.linspace( 0, 1, step_count)
        
        # Precompute Schauder bases
        self.schauder_bases = {
            "zero_boundary"      : None, 
            "shooting" : None
        }

        # Mode1 = Interpolation mode = Zero boundary functions
        basis = schauder_basis( self.n_max, self.time_grid.numpy() )
        basis = torch.t( torch.tensor( basis ) )
        basis = basis[:,1:] # Throw away the first basis vector, the constant function
        basis = basis[:,1:] # Throw away the second basis vector, the linear function
        N_max = 2**n_max-1  # Number of basis elements in Schauder basis is N_max = 2**n_max + 1 ; After throwing the two first basis vectors N_max = 2**n_max-1
        self.schauder_bases["zero_boundary"] = { "basis": basis, "N_max": N_max}

        # Mode2 = Shooting mode = Free endpoint
        basis = schauder_basis( self.n_max, self.time_grid.numpy() )
        basis = torch.t( torch.tensor( basis ) )
        basis = basis[:,1:] # Throw away the first basis vector, the constant function
        N_max = 2**n_max    # Number of basis elements in Schauder basis is N_max = 2**n_max + 1 ; After throwing the first basis vectors N_max = 2**n_max
        self.schauder_bases["shooting"] = { "basis": basis, "N_max": N_max}


    def computeGeodesicInterpolation(self, generator, m1, m2, epochs, optimizer_info, display_info) :
        """
            generator     : Function taking latent variables and generating a sample. Its gradient encodes the metric tensor.
            m1, m2        : Initial and destination point
            optimizer_info: dict containing
                            "name": Name of torch optimizer
                            "args": Learning rate, Momentum and Nesterov acceleration
            display_info  : String to add to progress bar, useful for identifying running task
        """
        N_max = self.schauder_bases["zero_boundary"]["N_max"]
        basis = self.schauder_bases["zero_boundary"]["basis"]
        # Dimension
        dim   = m1.shape[0]
        # parameters = Coefficients of (base+fiber) curve in Schauder basis
        parameters   = torch.zeros( (N_max, dim) , requires_grad=True)
        # Define linear interpolating curve == naive geodesic
        linear_curve = torch.ones( self.step_count, 1)*m1 + self.time_grid.view( self.step_count, 1)*(m2-m1)

        # Initialization
        curve  = linear_curve
        # Optimizer
        energy = 0
        optimizer = getattr( torch.optim, optimizer_info["name"] )
        optimizer = optimizer( [{ 'params': parameters }], **optimizer_info["args"] )
        # Loop
        with tqdm( range(epochs) ) as t:
            for i in t:
                # Compute curve of latent variables with suboptimal parameters
                curve             = linear_curve + torch.mm( basis, parameters )
                # Output
                generated_images  = generator( encoding = curve )
                # Finite difference computation of energy
                energy = (generated_images[1:,:]-generated_images[:-1,:]).pow(2).sum()
                # Optimize
                optimizer.zero_grad()
                energy.backward(retain_graph=True)
                grad_norm = parameters.grad.norm()
                t.set_description( "%s. Energy %f, Grad %f"%(display_info, energy, grad_norm) )
                optimizer.step()
            # End for
        # End with
        #
        # Recompute curve with optimal parameters
        geodesic_curve = linear_curve + torch.mm( basis, parameters )
        return linear_curve.detach().numpy(), geodesic_curve.detach().numpy()

    def computeGeodesicShooting(self, generator, b1, b2, f1, v1, epochs, optimizer_info, display_info):
        """
            generator     : Function taking latent variables and generating a sample. Its gradient encodes the metric tensor.
            b1, b2        : Initial and destination base point
            f1, v1        : Initial fiber point and fiber speed
            optimizer_info: dict containing
                            "name": Name of torch optimizer
                            "args": Learning rate, Momentum and Nesterov acceleration
            display_info  : String to add to progress bar, useful for identifying running task
        """
        N_max_1 = self.schauder_bases["zero_boundary"]["N_max"]
        basis1  = self.schauder_bases["zero_boundary"]["basis"]
        N_max_2 = self.schauder_bases["shooting"]["N_max"]
        basis2  = self.schauder_bases["shooting"]["basis"]
        # Dimensions
        base_dim     = b1.shape[0]
        fiber_dim    = f1.shape[0]
        # Initial (Naive) destination
        f2           = f1 + v1
        # parameters1: Coefficients of base curve in Schauder basis, with zero boundary condition
        # parameters2: Coefficients of fiber curve in Schauder basis, with free endpoint
        parameters1  = torch.zeros( (N_max_1, base_dim) , requires_grad=True)
        parameters2  = torch.zeros( (N_max_2, fiber_dim), requires_grad=True)
        # Define base curve and constant fiber curve
        linear_base_curve     = torch.ones ( self.step_count, 1)*b1 + self.time_grid.view( self.step_count, 1)*(b2-b1)
        linear_fiber_curve    = torch.ones ( self.step_count, 1)*f1 + self.time_grid.view( self.step_count, 1)*(f2-f1)
        linear_curve          = torch.cat( ( linear_fiber_curve, linear_base_curve), 1 )
        # Naive image obtained from naive transport
        endpoint    = linear_curve[-1,:]
        naive_image = generator( encoding = linear_curve[-1,:].view(-1, endpoint.shape[0]) )

        # Initialization
        base_curve   = linear_base_curve
        fiber_curve  = linear_fiber_curve
        # Optimizer
        energy     = 0
        lambda_reg = optimizer_info["lambda"]
        optimizer  = getattr( torch.optim, optimizer_info["name"] )
        optimizer  = optimizer( [{ 'params': parameters1 }, { 'params': parameters2 }], **optimizer_info["args"] )
        # Loop
        energy_series = []
        with tqdm( range(epochs) ) as t:
            for i in t:
                # Compute curves with suboptimal parameters
                base_curve        = linear_base_curve  + torch.mm( basis1, parameters1 )
                fiber_curve       = linear_fiber_curve + torch.mm( basis2, parameters2 )
                latent_variables  = torch.cat( (fiber_curve, base_curve), 1 )
                # Output
                generated_images  = generator( encoding = latent_variables )
                # Finite difference computation of energy
                energy = (generated_images[1:,:]-generated_images[:-1,:]).pow(2).sum()
                energy_series.append( energy.detach().numpy() )
                # Loss
                regularization = lambda_reg*(generated_images[-1,:]-naive_image).pow(2).sum()
                loss = energy + regularization
                # Optimize
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                grad1_norm = parameters1.grad.norm()
                grad2_norm = parameters2.grad.norm()
                t.set_description( "%s. Energy %f, Reg: %f, Base grad %f, Fiber grad %f"%(display_info, energy, regularization, grad1_norm, grad2_norm) )
                optimizer.step()
            # End for
        # End with
        energy_series = np.array( energy_series )
        #
        # Compute curves with optimal parameters
        naive_curve    = torch.cat( (linear_fiber_curve, linear_base_curve), 1)
        base_curve     = linear_base_curve  + torch.mm( basis1, parameters1 )
        fiber_curve    = linear_fiber_curve + torch.mm( basis2, parameters2 )
        geodesic_curve = torch.cat( (fiber_curve, base_curve), 1 )
        return naive_curve.detach().numpy(), geodesic_curve.detach().numpy(), energy_series