import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

def integrate_over_domain( basis_1, basis_2, nx, ny, in_domain ):
    x = np.linspace(0., 2.*np.pi, nx)
    y = np.linspace(0., 2.*np.pi, ny)
    dx = 2.*np.pi/nx
    dy = 2.*np.pi/ny
    xgrid, ygrid = np.meshgrid(x,y)
    domain_grid = in_domain(xgrid, ygrid)
    fn1 = ma.masked_where( ~domain_grid, basis_1(xgrid,ygrid))
    fn2 = ma.masked_where( ~domain_grid, basis_2(xgrid,ygrid))
    plt.pcolormesh(xgrid, ygrid, fn1)
    plt.show()
    plt.clf()
    value = np.sum( fn1*fn2 )*dx*dy
    return value

def generate_1D_basis_functions( nmax ):
    functions = []
    for n in range(nmax+1):
        functions.append( lambda x : np.cos(n*x) )
        functions.append( lambda x : np.sin(n*x) )
    return functions

def generate_2D_basis_functions( nmax):
    x_fns = generate_1D_basis_functions(nmax)
    y_fns = generate_1D_basis_functions(nmax)
    functions = []
    for xfn in x_fns:
        for yfn in y_fns:
            functions.append( lambda x,y: xfn(x)*yfn(y)/np.pi )
    return functions

def assemble_slepian_matrix( nmax, in_domain ):
    mat = np.empty( (2*(nmax+1)*2*(nmax+1), 2*(nmax+1)*2*(nmax+1) ) )
    print mat.shape

    nx = 30
    ny = 30

    i = 0
    n = 0
    m = 0

    basis1 = generate_2D_basis_functions(nmax)
    basis2 = generate_2D_basis_functions(nmax)

    print integrate_over_domain( basis1[10], basis1[10], nx, ny, in_domain)
    for i,b1 in enumerate(basis1):
        for j,b2 in enumerate(basis2):
            mat[i,j] = integrate_over_domain( b1, b2, nx, ny, in_domain )
    return mat
           

def in_domain( x, y ):
    r = 300.0
    return (x-np.pi)*(x-np.pi) + (y-np.pi)*(y-np.pi) <= r*r

mat = assemble_slepian_matrix( 2, in_domain )
print mat
