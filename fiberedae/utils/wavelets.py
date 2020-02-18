import numpy as np

# Mathematical source for both Haar and Schauder(-Faber) bases: https://en.wikipedia.org/wiki/Haar_wavelet

# Numpy implementation of Haar wavelet generation on [0,1]
def haar_wavelet(time_grid):
    index_set1 = np.where( (0   <= time_grid) & (time_grid < 0.5) )
    index_set2 = np.where( (0.5 <= time_grid) & (time_grid < 1  ) )
    wavelet = np.zeros_like( time_grid )
    wavelet[ index_set1 ] = 1
    wavelet[ index_set2 ] = -1
    return wavelet

def haar_basis_element(n, k, time_grid):
    return (2.0**(0.5*n)) * haar_wavelet( (2**n)*time_grid- k)


# Returns a total of 2**n_max basis functions corresponding to
#   the Haar wavelets (n,k) corresponding to
#   n=0, ..., n_max-1 and 
#   k=0, ..., 2**n - 1
def haar_basis( n_max, time_grid):
    basis = []
    # Constant function
    basis.append( np.ones_like(time_grid) )
    # The next ones (n>0)
    for n in range(n_max):
        basis = basis + [ haar_basis_element(n,k, time_grid) for k in range(2**n)]
    return basis
    

# Numpy implementation of Schauder basis generation on [0,1]

# Schauder elements are integrals of the Haar elements
def schauder_basis_element(n, k, time_grid):
    haar_element =  haar_basis_element(n, k, time_grid)
    element = np.zeros_like( haar_element )
    element[1:] = (2**(1+0.5*n)) * 0.5* ( haar_element[:-1] + haar_element[1:] )* (time_grid[1:] - time_grid[:-1])
    return element.cumsum()

def schauder_basis( n_max, time_grid):
    basis = []
    # Constant function
    basis.append( np.ones_like(time_grid) )
    # Identity map
    basis.append( np.copy(time_grid) )
    # The next ones (n>0)
    for n in range(n_max):
        basis = basis + [ schauder_basis_element(n,k, time_grid) for k in range(2**n)]
    return basis