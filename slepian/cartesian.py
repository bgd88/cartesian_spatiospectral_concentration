from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import numpy.ma as ma
import numpy.linalg as linalg

def integrate_over_domain( basis_1, basis_2, nx, ny, in_domain ):
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    dx = 2.*np.pi/nx
    dy = 2.*np.pi/ny
    xgrid, ygrid = np.meshgrid(x,y)
    domain_grid = in_domain(xgrid, ygrid)
    fn1 = ma.masked_where( ~domain_grid, basis_1(xgrid,ygrid))
    fn2 = ma.masked_where( ~domain_grid, basis_2(xgrid,ygrid))
    value = np.sum( fn1*fn2 )*dx*dy
    return value
    
def generate_1D_basis_functions():
    yield lambda x: np.ones_like(x)/np.sqrt(2.*np.pi)
    n = 1
    while True:
        yield lambda x: np.cos(n*x)/np.sqrt(np.pi)
        yield lambda x: np.sin(n*x)/np.sqrt(np.pi)
        n += 1

def generate_2D_basis_functions(nmax):
    genX = generate_1D_basis_functions()
    for ii in range(2*nmax + 1):
        genY = generate_1D_basis_functions()
        xfn = genX.next()
        for jj in range(2*nmax + 1):
            yfn = genY.next()
            yield lambda x, y : xfn(x)*yfn(y)
    raise StopIteration

def assemble_slepian_matrix( nmax, in_domain ):
    mat = np.empty( ((2*nmax+1)**2, (2*nmax+1)**2) )
    nx, ny = (10*nmax, 10*nmax) # 10 quadrature points per wavelegth
    gen1 = generate_2D_basis_functions(nmax)
    for ii in range((2*nmax+1)**2):
        gen2 = generate_2D_basis_functions(nmax)
        b1 = gen1.next()
        for jj in range((2*nmax+1)**2):
            b2 = gen2.next()
            mat[ii, jj] = integrate_over_domain( b1, b2, nx, ny, 
                                                 in_domain )
    return mat
           
def reconstruct_eigenvectors(eigenvecs, eigenvals, nmax, shannon, nx=100, ny=100):

    # Shannon number cannot be larger than n_modes:
    n_modes = (2*nmax+1)**2
    assert (shannon < n_modes)

    # Sort by the largest eigenvalues
    idx = eigenvals.argsort()[::-1]   
    sorted_eigenvals = eigenvals[idx]
    sorted_eigenvecs = eigenvecs[:,idx]

    # Setup the grid for evaluating the functions
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    xgrid, ygrid = np.meshgrid(x,y)

    solution = []

    for i in range(shannon):
        vec = sorted_eigenvecs[:,i]
        gen = generate_2D_basis_functions(nmax)
        slepian_function = np.zeros_like(xgrid)
        for j in range(n_modes):
            try:
                fn = gen.next()
                slepian_function += vec[j]*fn(xgrid,ygrid)
            except StopIteration:
                raise Exception("Mismatch between expected length of an eigenvector and its actual length")
        solution.append( (sorted_eigenvals[i], slepian_function) )
    return solution

def compute_slepian_basis( nmax, in_domain ):
    print("Assembling matrix")
    mat = assemble_slepian_matrix( nmax, in_domain )
    print("Solving eigenvalue problem")
    eigenvals,eigenvecs = linalg.eigh(mat)
    print("Reconstructing eigenvectors")
    shannon = int(1.5*np.ceil(nmax*nmax*in_domain.area/4./np.pi))
    basis = reconstruct_eigenvectors( eigenvecs, eigenvals, nmax, shannon)
    return basis
