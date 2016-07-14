import numpy as np
import matplotlib.pyplot as plt
import slepian

def in_domain( x, y ):
    r = np.pi
    return (x-np.pi)*(x-np.pi) + (y-np.pi)*(y-np.pi) <= r*r
in_domain.area = np.pi*np.pi*np.pi

nx,ny = 100,100
x = np.linspace(0, 2*np.pi, nx)
y = np.linspace(0, 2*np.pi, ny)
xgrid, ygrid = np.meshgrid(x,y)

basis = slepian.compute_slepian_basis( 2, in_domain )

for (eigenvalue, function) in basis:
    print("Slepian basis function with eigenvalue : ",eigenvalue)
    cm = plt.pcolormesh(xgrid,ygrid,function, cmap='RdBu', lw=0)
    plt.colorbar()
    c = plt.Circle( (np.pi,np.pi), 3., color='k', fill=False)
    plt.gca().add_artist(c)
    plt.show()
    plt.clf()

