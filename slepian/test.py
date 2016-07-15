import numpy as np
import pickle
import slepian 

def check_orthogonality(domain, basis, tol=3e-2):
    # TODO: Think of something for nx and ny
    dx = domain.extent[0]/1000.
    dy = domain.extent[1]/1000.
    max_error = 0
    for i, (eigVal_i, eigFun_i) in enumerate(basis):
        for j, (eigVal_j, eigFun_j) in enumerate(basis):
            inner_product = np.sum(eigFun_i*eigFun_j)*dx*dy
            print( "i = {}, j = {}, <b_i, b_j> = {}".format(i,j, inner_product) )
            if i != j:
                max_error = max(inner_product, max_error)
                assert np.abs(inner_product) < tol, "Not orthogonal"
            else:
                max_error = max(inner_product-1., max_error)
                assert np.abs(inner_product-1) < tol, "Not orthonormal"
    print("Slepian Basis is orthonormal within a toleraance of {}".format(tol))
    print("Maximum Error: {}".format(max_error))
