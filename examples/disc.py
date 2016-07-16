import sys
import numpy as np
import matplotlib.pyplot as plt
import sleppy

R = 2.
domain = sleppy.Disc( (2.*np.pi, 2.*np.pi), (np.pi,np.pi), R)

nx,ny = 100,100
x = np.linspace(0, 2*np.pi, nx)
y = np.linspace(0, 2*np.pi, ny)
xgrid, ygrid = np.meshgrid(x,y)

basis = sleppy.compute_slepian_basis( domain, 4)

for (eigenvalue, function) in basis:
    print("Slepian basis function with eigenvalue : ",eigenvalue)
    cm = plt.pcolormesh(xgrid,ygrid,function(xgrid, ygrid), cmap='RdBu', lw=0)
    plt.colorbar(cm)
    c = plt.Circle( (np.pi,np.pi), R, color='k', fill=False)
    plt.gca().add_artist(c)
    plt.show()
    plt.clf()
