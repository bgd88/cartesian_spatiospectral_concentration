import numpy as np
import pickle


def check_orthogonality(domain, basis, tol=3e-2, nx=500, ny=500):
    # Build Grids
    x = np.linspace(0, domain.extent[0], nx)
    y = np.linspace(0, domain.extent[1], ny)
    xgrid, ygrid = np.meshgrid(x, y)
    dx = float(domain.extent[0]) / nx
    dy = float(domain.extent[1]) / ny
    max_error = 0
    for i, (eigVal_i, eigFun_i) in enumerate(basis):
        for j, (eigVal_j, eigFun_j) in enumerate(basis):
            inner_product = np.sum(eigFun_i(xgrid, ygrid)
                                   * eigFun_j(xgrid, ygrid)) * dx * dy
            if i != j:
                max_error = max(inner_product, max_error)

                assert np.abs(inner_product) < tol, \
                    " = {}, j = {}, <b_i, b_j> = {} - Not orthonormal".format(
                        i, j, inner_product)

            else:
                max_error = max(inner_product - 1., max_error)
                assert np.abs(inner_product - 1) < tol, \
                    " = {}, j = {}, <b_i, b_j> = {} - Not orthonormal".format(
                        i, j, inner_product)
    print("Slepian Basis is orthonormal within a toleraance of {}".format(tol))
    print("Maximum Error: {}".format(max_error))
