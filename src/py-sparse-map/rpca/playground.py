import numpy as np
import sklearn.datasets, sklearn.decomposition
from util import *

def shrink(x, tau):
    return np.sign(x) * np.maximum(np.abs(x)-tau, 0)


def make_random_noise(x, rate, amplitude=10.0):
    noise = np.random.random(x.shape)
    noise[noise < (1.0-rate)] = 0
    noise[noise >= (1.0-rate)] = amplitude
    return x + noise

np.random.seed(17)

data = sklearn.datasets.load_iris()
X = data.data
Morig = np.mean(X, 0)

X = make_random_noise(X, 0.005)

batch_size, input_size = X.shape

nComp = 3
tol = 1e-06
lam = 1.0/np.sqrt(np.max(X.shape))
mu = 2.0*np.mean(Morig)

L = X.copy()
S = np.zeros((batch_size, input_size))
Y = np.zeros((batch_size, input_size))

for it in xrange(1):
    Lw = X - S + (1.0/mu)*Y
    
    M = np.mean(Lw, axis=0)
    C = np.cov(Lw.T)
    eigval, eigvec = np.linalg.eig(C)
    PC = np.dot(Lw - M, eigvec)

    L = M + np.dot(PC[:,:nComp], eigvec[:, :nComp].T)

    S = shrink(X - L + (1.0/mu)*Y, tau=1.0) #tau=lam/mu)

    Z = X - L - S

    Y = Y + mu*Z
    err = np.linalg.norm(Z)

    if it % 10 == 0:
        print "Iteration {}, error: {}".format(it, err)

    if err < tol: 
        break

# shs(
#     PC[np.where(data.target == 0)[0], :nComp],
#     PC[np.where(data.target == 1)[0], :nComp],
#     PC[np.where(data.target == 2)[0], :nComp],
#     labels=["red", "blue", "green"],
#     show=True    
# )