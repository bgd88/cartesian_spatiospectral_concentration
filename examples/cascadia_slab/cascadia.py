import numpy as np
import matplotlib.pyplot as plt
import sleppy
import matplotlib.patches as patches
import sleppy.test as test

nmax = 4
perimeter_file = "cas_slab1.0.clip"
out_file = "cascadia_basis_nmax{}.pkl".format(nmax)
domain = sleppy.LatLonFromPerimeterFile(perimeter_file)

nx, ny = 100, 100
x = np.linspace(0, domain.extent[0], nx)
y = np.linspace(0, domain.extent[1], ny)
xgrid, ygrid = np.meshgrid(x, y)

basis = sleppy.compute_slepian_basis(
    domain, nmax, basis_function_type='interpolated')
test.check_orthogonality(domain, basis)
for (eigenvalue, function) in basis:
    print("Slepian basis function with eigenvalue : ", eigenvalue)
    cm = plt.pcolormesh(xgrid, ygrid, function(
        xgrid, ygrid), cmap='RdBu', lw=0)
    plt.colorbar(cm)
    patch = patches.PathPatch(domain.perimeter_path, facecolor='none', lw=2)
    plt.gca().add_patch(patch)
    plt.title("Slepian basis function with eigenvalue : {}".format(eigenvalue))
    plt.show()
    plt.clf()
