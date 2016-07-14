import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

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
    
def generate_1D_basis_functions( ):
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
    nx, ny = (300, 300)
    gen1 = generate_2D_basis_functions(nmax)
    for ii in range((2*nmax+1)**2):
        gen2 = generate_2D_basis_functions(nmax)
        b1 = gen1.next()
        for jj in range((2*nmax+1)**2):
            b2 = gen2.next()
            mat[ii, jj] = integrate_over_domain( b1, b2, nx, ny, 
                                             in_domain )
    return mat
           
def in_domain( x, y ):
    r = 300.0
    return (x-np.pi)*(x-np.pi) + (y-np.pi)*(y-np.pi) <= r*r

mat = assemble_slepian_matrix( 2, in_domain )
for i in range(mat.shape[0]):
    print mat[i,i]
plt.imshow(mat)
plt.show()

