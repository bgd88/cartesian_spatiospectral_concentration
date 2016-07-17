import numpy as np
import matplotlib.pyplot as plt
import sleppy
import sleppy.test as test

domain = sleppy.Rectangle( (10., 10.), (2., 2.), (8.,8.))

nx,ny = 100,100
x = np.linspace(0, 10., nx)
y = np.linspace(0, 10., ny)
xgrid, ygrid = np.meshgrid(x,y)

basis = sleppy.compute_slepian_basis( domain, 4 , basis_function_type='interpolated')
test.check_orthogonality(domain, basis)
for (eigenvalue, function) in basis:
    print("Slepian basis function with eigenvalue : ",eigenvalue)
    cm = plt.pcolormesh(xgrid,ygrid,function(xgrid,ygrid), cmap='RdBu', lw=0)
    plt.colorbar(cm)
    r = plt.Rectangle( (2.,2.), 6.,6., fill=False)
    plt.gca().add_artist(r)
    plt.show()
    plt.clf()

