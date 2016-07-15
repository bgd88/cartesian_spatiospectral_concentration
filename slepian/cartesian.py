from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import numpy.ma as ma
import numpy.linalg as linalg
import multiprocessing 

def integrate_over_domain( domain, basis_1, basis_2, nx, ny):
    x = np.linspace(0, domain.extent[0], nx)
    y = np.linspace(0, domain.extent[1], ny)
    dx = domain.extent[0]/nx
    dy = domain.extent[1]/ny
    xgrid, ygrid = np.meshgrid(x,y)
    domain_grid = domain.in_subdomain(xgrid, ygrid)
    fn1 = ma.masked_where( ~domain_grid, basis_1(xgrid,ygrid))
    fn2 = ma.masked_where( ~domain_grid, basis_2(xgrid,ygrid))
    value = np.sum( fn1*fn2 )*dx*dy
    return value
    
def generate_1D_basis_functions( length ):
    yield lambda x: np.ones_like(x)/np.sqrt(length)
    n = 1
    while True:
        yield lambda x: np.cos(n*np.pi*x/length)/np.sqrt(length/2.)
        yield lambda x: np.sin(n*np.pi*x/length)/np.sqrt(length/2.)
        n += 1

def generate_2D_basis_functions(nmax, width, height):
    genX = generate_1D_basis_functions(width)
    for ii in range(2*nmax + 1):
        genY = generate_1D_basis_functions(height)
        xfn = next(genX)
        for jj in range(2*nmax + 1):
            yfn = next(genY)
            yield lambda x, y : xfn(x)*yfn(y)
    raise StopIteration

def assemble_slepian_matrix(domain, nmax, numProc):
    assert type(numProc) == int, "Pretty sure you should have an integer number of processors"
    N = (2*nmax+1)**2
    ii = 0
    count = 0
    index_ranges = []
    mat = np.empty( (N, N) )
    while count < numProc:
        jj = min(np.ceil(np.sqrt(N**2/numProc) + ii**2 ).astype(int), N-1)
        index_ranges.append([ii, jj])
        ii = jj+1
        count +=1 
    jobs = []
    for index_range in index_ranges:
#        assemble_slepian_matrix_block(mat, domain, index_range, nmax)
        print([index_range, N])
        proc = multiprocessing.Process(
                target=assemble_slepian_matrix_block, 
                args=(mat, domain, index_range, nmax) )
        jobs.append(proc)
        proc.start()
    
    for j in jobs:
        j.join()
    
    # Handle the symmetry of the matrix
    for ii in range(N):
        for jj in range(ii, N):
            mat[ii,jj] = mat[jj,ii]
    return mat
        
def assemble_slepian_matrix_block(mat, domain, index_range, nmax):
    nx, ny = (10*nmax, 10*nmax) # 10 quadrature points per wavelegth
    gen1 = generate_2D_basis_functions(nmax, domain.extent[0], domain.extent[1])
    for ii in range((2*nmax+1)**2):
        b1 = next(gen1)
        if (ii >= index_range[0]) and (ii <= index_range[1]):
            gen2 = generate_2D_basis_functions(nmax, domain.extent[0], domain.extent[1])
            for jj in range((2*nmax+1)**2):
                b2 = next(gen2)
                if jj <= ii:  ## Not necessary to calculate the others due to symmetry
                    mat[ii, jj] = integrate_over_domain( domain, b1, b2, nx, ny)
           
def reconstruct_eigenvectors(domain, eigenvecs, eigenvals, nmax, cutoff=0.5, nx=100, ny=100):
    n_modes = (2*nmax+1)**2

    # Sort by the largest eigenvalues
    idx = eigenvals.argsort()[::-1]   
    sorted_eigenvals = eigenvals[idx]
    sorted_eigenvecs = eigenvecs[:,idx]
    cutoff_n = np.argmin( np.abs(cutoff - sorted_eigenvals/sorted_eigenvals[0]))

    # Setup the grid for evaluating the functions
    x = np.linspace(0, domain.extent[0], nx)
    y = np.linspace(0, domain.extent[1], ny)
    xgrid, ygrid = np.meshgrid(x,y)

    solution = []

    for i in range(cutoff_n):
        vec = sorted_eigenvecs[:,i]
        gen = generate_2D_basis_functions(nmax, domain.extent[0], domain.extent[1])
        slepian_function = np.zeros_like(xgrid)
        for j in range(n_modes):
            try:
                fn = next(gen)
                slepian_function += vec[j]*fn(xgrid,ygrid)
            except StopIteration:
                raise Exception("Mismatch between expected length of an eigenvector and its actual length")
        # Normalize solution
        slepian_function /= np.sqrt(np.sum(slepian_function*slepian_function)*domain.extent[0]*domain.extent[1]/nx/ny)
        solution.append( (sorted_eigenvals[i], slepian_function) )
    return solution

def compute_slepian_basis( domain, nmax, numProc=1):
    print("Assembling matrix")
    mat = assemble_slepian_matrix( domain, nmax, numProc )
    print("Solving eigenvalue problem")
    eigenvals,eigenvecs = linalg.eigh(mat)
    print("Reconstructing eigenvectors")
    shannon = int(1.5*np.ceil(np.pi*nmax*nmax*domain.area/domain.extent[0]/domain.extent[1]))
    basis = reconstruct_eigenvectors( domain, eigenvecs, eigenvals, nmax, cutoff=0.5)
    return basis
