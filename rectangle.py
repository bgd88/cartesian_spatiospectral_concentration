import numpy as np
import matplotlib.pyplot as plt
import slepian

domain = slepian.Rectangle( (10., 10.), (2., 2.), (8.,8.))

nx,ny = 100,100
x = np.linspace(0, 10., nx)
y = np.linspace(0, 10., ny)
xgrid, ygrid = np.meshgrid(x,y)

basis = slepian.compute_slepian_basis( domain, 6 )

for (eigenvalue, function) in basis:
    print("Slepian basis function with eigenvalue : ",eigenvalue)
    cm = plt.pcolormesh(xgrid,ygrid,function, cmap='RdBu', lw=0)
    plt.colorbar(cm)
    r = plt.Rectangle( (2.,2.), 6.,6., fill=False)
    plt.gca().add_artist(r)
    plt.show()
    plt.clf()

