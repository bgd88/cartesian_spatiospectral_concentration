from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import numpy.ma as ma
import numpy.linalg as linalg
from numpy.polynomial.legendre import leggauss
import multiprocessing 
from scipy.interpolate import LinearNDInterpolator as linear_interp
import copy 

def integrate_over_domain( domain, basis_1, basis_2, x_degree, y_degree):
    # Get the gauss-legendre points and weights
    x, xw = leggauss( x_degree )
    y, yw = leggauss( y_degree )
    # Rescale them to our domain interval
    x = (x+1.)*domain.extent[0]/2.
    y = (y+1.)*domain.extent[1]/2.
    xw *= domain.extent[0]/2.
    yw *= domain.extent[1]/2.
    # Make the grid on which we will compute the integral
    xgrid, ygrid = np.meshgrid(x,y)
    weights = np.outer(yw,xw)

    # Integrate basis_1*basis_2 over the grid
    domain_grid = domain.in_subdomain(xgrid, ygrid)
    fn1 = ma.masked_where( ~domain_grid, basis_1(xgrid,ygrid))
    fn2 = ma.masked_where( ~domain_grid, basis_2(xgrid,ygrid))
    value = np.sum( fn1*fn2*weights )

    return value

    
def generate_1D_basis_functions( length ):
    yield lambda x: np.ones_like(x)/np.sqrt(length)
    n = 1
    while True:
        yield lambda x: np.cos(2.*n*np.pi*x/length)/np.sqrt(length/2.)
        yield lambda x: np.sin(2.*n*np.pi*x/length)/np.sqrt(length/2.)
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
    # Allocate room for the slepian matrix
    mat = np.empty( (N, N) )

    # Partition the rows among the processes,
    # where we attempt to have each process
    # get approximately the same number of entries
    # to calculate. Since the matrix is symmetric,
    # each process gets a different trapezoidal region
    # of the overall matrix
    ii = 0
    count = 0
    index_ranges = []
    while count < numProc:
        jj = min(np.ceil( np.sqrt(N**2/numProc + ii**2) ).astype(int), N-1)
        index_ranges.append([ii, jj])
        ii = jj+1
        count +=1

    # Spin up the processes
    jobs = [] # list of processes
    pipes = [] # list of pipes for communication
    for index_range in index_ranges:
        c_recv, c_send = multiprocessing.Pipe(duplex=False)
        proc = multiprocessing.Process(
                target=assemble_slepian_matrix_block, 
                args=(c_send, domain, index_range, nmax) )
        pipes.append(c_recv)
        jobs.append(proc)
        proc.start()

    # Get the submatrices calculated in each of the pipes
    # and fill the main matrix with them.
    for conn, index_range in zip(pipes, index_ranges):
        submatrix = conn.recv()[0]
        mat[index_range[0]:index_range[1]+1, :] = submatrix

    # Close the subprocesses
    for j in jobs:
        j.join()
    
    # Handle the symmetry of the matrix
    for ii in range(N):
        for jj in range(ii, N):
            mat[ii,jj] = mat[jj,ii]
    return mat

        
def assemble_slepian_matrix_block(conn, domain, index_range, nmax):
    N = (2*nmax+1)**2

    # Allocate the submatrix
    submatrix = np.empty( (index_range[1]-index_range[0]+1, N) )
    nx, ny = (10*nmax, 10*nmax) # 10 quadrature points per wavelegth

    gen1 = generate_2D_basis_functions(nmax, domain.extent[0], domain.extent[1])
    for ii in range(N):
        b1 = next(gen1)
        if (ii >= index_range[0]) and (ii <= index_range[1]):
            gen2 = generate_2D_basis_functions(nmax, domain.extent[0], domain.extent[1])
            for jj in range(N):
                b2 = next(gen2)
                if jj <= ii:  ## Not necessary to calculate the others due to symmetry
                    submatrix[ii-index_range[0], jj] = integrate_over_domain( domain, b1, b2, nx, ny)
    conn.send([submatrix,])


def parallel_reconstruct_basis_functions(conn, index_list, sorted_eigenvecs, domain, nmax, basis_function_type):
    solution, indices = [], []
    for i in index_list:
        spectral_coefs = sorted_eigenvecs[:, i]
        if basis_function_type is 'interpolated':
            # Create interpolator function
            slepian_function = interpolated_slepian_basis_function(domain, spectral_coefs, nmax)
        elif basis_function_type is 'exact':
            # Create function using spectral coefficients
            slepian_function = slepian_basis_function(domain, spectral_coefs, nmax)
        solution.append(slepian_function)
        indices.append(i)
    conn.send([solution, indices])


def reconstruct_eigenvectors(domain, eigenvecs, eigenvals, nmax, numProc, cutoff=0.5, nx=100, ny=100,
                             basis_function_type='exact'):
    n_modes = (2*nmax+1)**2

    # Sort by the largest eigenvalues
    idx = eigenvals.argsort()[::-1]   
    sorted_eigenvals = eigenvals[idx]
    sorted_eigenvecs = eigenvecs[:,idx]
    cutoff_n = np.argmin( np.abs(cutoff - sorted_eigenvals/sorted_eigenvals[0]))
    
    # Spin up the processes
    jobs = [] # list of processes
    pipes = [] # list of pipes for communication
    indices = range(cutoff_n)
    partitioned_indices = [ indices[i::numProc] for i in range(numProc) ]
    
    for index_list in partitioned_indices:
        c_recv, c_send = multiprocessing.Pipe(duplex=False)
        proc = multiprocessing.Process(target=parallel_reconstruct_basis_functions,
                args=(c_send, index_list, sorted_eigenvecs, domain, nmax, basis_function_type) )
        pipes.append(c_recv)
        jobs.append(proc)
        proc.start()
        
    # Get the resulting eigenfunctions in each pipe
    solution, solution_indices = [], []
    for conn in pipes:
        slepian_functions, indices = conn.recv()
        solution_indices += indices
        solution += slepian_functions 
    idx = np.array(solution_indices).argsort()
    sorted_solution = list(np.array(solution)[idx])

    for j in jobs:
        j.join()
    
    return [(eig, fun) for eig, fun in  zip(sorted_eigenvals, sorted_solution)]


def compute_slepian_basis( domain, nmax, numProc=multiprocessing.cpu_count(), basis_function_type='exact'):
    print("Assembling matrix")
    mat = assemble_slepian_matrix( domain, nmax, numProc )
    print("Solving eigenvalue problem")
    eigenvals, eigenvecs = linalg.eigh(mat)
    print("Reconstructing eigenvectors")
    shannon = int(1.5*np.ceil(np.pi*nmax*nmax*domain.area/domain.extent[0]/domain.extent[1]))
    basis = reconstruct_eigenvectors(domain, eigenvecs, eigenvals, nmax, numProc,
                                     basis_function_type=basis_function_type)
    return basis


class slepian_basis_function(object):
    "Create Slepian Basis Function from spectral coefficients."
    def __init__(self, domain, spectral_coefs, nmax):
        self.spectral_coefs = spectral_coefs
        self.extent = domain.extent
        self.nmax = nmax
        self.nmodes = (2*nmax + 1)**2
        # Check specral coeficients
        error_message = "nmax not consistent with the number of spectral coefficients."
        assert self.nmodes == self.spectral_coefs.size, error_message
        self._set_normalization_coef()

    def _set_normalization_coef(self):
        self.normalization_coef = np.sqrt(np.sum(self.spectral_coefs*self.spectral_coefs))

    def __call__(self, x, y):
        gen = generate_2D_basis_functions(self.nmax, self.extent[0], self.extent[1])
        slepian_grid_values = np.zeros_like(x)
        for coef in self.spectral_coefs:
            try:
                fn = next(gen)
                slepian_grid_values += coef*fn(x,y)
            except StopIteration:
                raise Exception("Mismatch between expected length of an eigenvector and its actual length")
        return slepian_grid_values/self.normalization_coef

class interpolated_slepian_basis_function(slepian_basis_function):
    def __init__(self, domain, spectral_coefs, nmax, nx=100, ny=100):
        super(interpolated_slepian_basis_function, self).__init__(domain, spectral_coefs, nmax)
        self.interpolator = self._create_interpolator(nx, ny) 
    
    def _create_interpolator(self, nx, ny): 
        # Setup the grid for evaluating the functions
        x = np.linspace(0, self.extent[0], nx)
        y = np.linspace(0, self.extent[1], ny)
        xgrid, ygrid = np.meshgrid(x,y)
        dx = self.extent[0]/nx
        dy = self.extent[1]/ny
        
        slepian_grid_values = super(interpolated_slepian_basis_function, self).__call__(xgrid, ygrid) 
        # Create linear interpolator
        pts, vals = [p for p in zip(xgrid.flatten(), ygrid.flatten())], slepian_grid_values.flatten()
        return linear_interp(pts, vals)
    
    def __call__(self, x, y):
        return self.interpolator(x, y) 
