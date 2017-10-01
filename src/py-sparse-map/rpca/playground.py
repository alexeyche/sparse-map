import numpy as np
import sklearn.datasets, sklearn.decomposition
from util import *
from rpca.util import *

def reconstruct_pca(x, ncomp):
    M = np.mean(x, axis=0)
    C = np.cov(x.T)
    eigval, eigvec = np.linalg.eig(C)
    PC = np.dot(x - M, eigvec)

    L = M + np.dot(PC[:,:ncomp], eigvec[:, :ncomp].T)

    return L, PC, eigval, eigvec


np.random.seed(17)

# data = sklearn.datasets.load_iris()
# X = data.data
# Morig = np.mean(X, 0)

# X = make_random_noise(X, 0.005)


X, X0, S0 = generate_toy_data(50, 100, 5, 0.2)
normX = np.linalg.norm(X, "fro")


batch_size, input_size = X.shape

ncomp = 5
tol = 1e-06
lam = 1.0/np.sqrt(np.max(X.shape))
mu = 10.0 * lam
# mu = 2.0*np.mean(Morig)


L = np.zeros((batch_size, input_size))
S = np.zeros((batch_size, input_size))
Y = np.zeros((batch_size, input_size))


# L, PC, eigval, eigvec = reconstruct_pca(X, ncomp)


for it in xrange(100):
    # L = singular_shrink(X - S + (1.0/mu)*Y, tau=1.0/mu)

    L, PC, eigval, eigvec = reconstruct_pca(X - S + (1.0/mu)*Y, ncomp)
    S = shrink(X - L + (1.0/mu)*Y, tau=lam/mu)

    Z = X - L - S

    Y = Y + mu*Z
    err = np.linalg.norm(Z, "fro")/normX

    if it % 10 == 0:
        print "Iteration {}, error: {}".format(it, err)

    if err < tol: 
        break

# shs(
#     PC[np.where(data.target == 0)[0], :ncomp],
#     PC[np.where(data.target == 1)[0], :ncomp],
#     PC[np.where(data.target == 2)[0], :ncomp],
#     labels=["red", "blue", "green"],
#     show=True    
# )