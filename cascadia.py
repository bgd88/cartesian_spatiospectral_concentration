import numpy as np
import matplotlib.pyplot as plt
import slepian
import matplotlib.patches as patches
import pickle

perimeter_file = "examples/cas_slab1.0.clip"
out_file = "cascadia_basis.pkl"
domain = slepian.LatLonFromPerimeterFile(perimeter_file)

nx,ny = 100,100
x = np.linspace(0, domain.extent[0], nx)
y = np.linspace(0, domain.extent[1], ny)
xgrid, ygrid = np.meshgrid(x,y)

basis = slepian.compute_slepian_basis( domain, 6)
with open(out_file, 'wb') as f:
    pickle.dump([domain, basis, xgrid, ygrid], f)

count = 0
for (eigenvalue, function) in basis:
    print("Slepian basis function with eigenvalue : ",eigenvalue)
    cm = plt.pcolormesh(xgrid,ygrid,function, cmap='RdBu', lw=0)
    plt.colorbar(cm)
    patch = patches.PathPatch(domain.perimeter_path, facecolor='none', lw=2)
    plt.gca().add_patch(patch)
    plt.title( "Slepian basis function with eigenvalue : {}".format(eigenvalue) )
    plt.savefig("figures/{}_eigenfunction.png".format(count), bbox_inches='tight', dpi=250)
    count += 1
    plt.clf()
