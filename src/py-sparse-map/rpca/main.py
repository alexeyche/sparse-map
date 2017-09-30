

from util import *
from sklearn.datasets import load_iris
import tensorflow as tf
import numpy as np
from numpy.linalg import svd
from ts_pp import generate_ts

def shrink(x, tau):
    return np.sign(x) * np.maximum(np.abs(x)-tau, 0)

def make_random_noise(x, rate, amplitude=5.0):
    noise = np.random.random(x.shape)
    noise[noise < (1.0-rate)] = 0
    noise[noise >= (1.0-rate)] = amplitude
    return x + noise

def singular_shrink(x, tau):
    U, S, V = svd(x, full_matrices=False)
    # print "Shrink with tau: {}".format(tau) 
    # print "\t", S
    # Ss = shrink(S, tau)
    # print "\t", Ss
    # print "Non zero entries: {}".format(len(np.where(Ss != 0.0)[0]))
    # print "==="
    return np.dot(U, np.dot(np.diag(shrink(S, tau)), V))



np.random.seed(10)

seq_size = 400

x_v = np.asarray([
    generate_ts(seq_size)
    for _ in xrange(10)
]).T
x_v = make_random_noise(x_v, 0.001)

lam = 1.0/np.sqrt(np.max(x_v.shape))
mu = 0.01 * lam

tol=1e-06
max_iter=1000

# C = np.cov(x_v.T)
# eigval, eigvec = np.linalg.eig(C)
# PC = np.dot(x_v, eigvec)[:,0:2]

batch_size, input_size = x_v.shape

X = x_v

L = np.zeros((batch_size, input_size))
S = np.zeros((batch_size, input_size))
Y = np.zeros((batch_size, input_size))

for it in xrange(1000):
    L = singular_shrink(X - S + (1.0/mu)*Y, tau=1.0/mu)
    S = shrink(X - L + (1.0/mu)*Y, tau=lam/mu)
    # print np.mean(S)
    Z = X - L - S

    Y = Y + mu*Z
    err = np.linalg.norm(Z)

    if it % 10 == 0:
        print "Iteration {}, error: {}".format(it, err)

    if err < tol: 
        break

# dd = L - X
# eigval_dd, eigvec_dd = np.linalg.eig(np.cov((L - X).T))

# PC = np.dot(x_v, eigvec)[:,0:2]
# PC_dd = np.dot(x_v, eigvec_dd)[:,0:2]


# shs(
#     PC_dd[np.where(data.target == 0)],
#     PC_dd[np.where(data.target == 1)],
#     PC_dd[np.where(data.target == 2)],
#     labels=["red", "blue", "green"],
#     show=False    
# )

# shs(
#     PC[np.where(data.target == 0)],
#     PC[np.where(data.target == 1)],
#     PC[np.where(data.target == 2)],
#     labels=["red", "blue", "green"],
#     show=True    
# )