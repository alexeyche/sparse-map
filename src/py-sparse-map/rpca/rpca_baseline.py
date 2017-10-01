

from util import *
from sklearn.datasets import load_iris
import tensorflow as tf
import numpy as np
from ts_pp import generate_ts
from rpca.util import shrink, singular_shrink, generate_toy_data



np.random.seed(12)

X, X0, S0 = generate_toy_data(50, 100, 5, 0.2)


normX = np.linalg.norm(X, "fro")

lam = 1.0/np.sqrt(np.max(X.shape))
mu = 10.0 * lam

tol=1e-06
max_iter=1000

# C = np.cov(x_v.T)
# eigval, eigvec = np.linalg.eig(C)
# PC = np.dot(x_v, eigvec)[:,0:2]

batch_size, input_size = X.shape

L = np.zeros((batch_size, input_size))
S = np.zeros((batch_size, input_size))
Y = np.zeros((batch_size, input_size))

for it in xrange(200):
    L = singular_shrink(X - S + (1.0/mu)*Y, tau=1.0/mu)
    S = shrink(X - L + (1.0/mu)*Y, tau=lam/mu)
    # print np.mean(S)
    Z = X - L - S

    Y = Y + mu*Z
    err = np.linalg.norm(Z, "fro")/normX

    if it % 10 == 0:
        print "Iteration {}, error: {}".format(it, err)

    if err < tol: 
        break
