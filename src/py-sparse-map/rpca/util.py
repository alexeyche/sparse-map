import numpy as np
from numpy.linalg import svd

from matplotlib import pyplot as plt


def shrink(x, tau):
    return np.sign(x) * np.maximum(np.abs(x)-tau, 0)


def singular_shrink(x, tau, n=None):
    U, S, V = svd(x, full_matrices=False)

    # print "Shrink with tau: {}".format(tau) 
    # print "\t", S
    # Ss = shrink(S, tau)
    # print "\t", Ss
    # print "Non zero entries: {}".format(len(np.where(Ss != 0.0)[0]))
    # print "==="
    if n is None:
        Ss = shrink(S, tau)

        # shl(Ss, S)
        # print np.mean(np.square(Ss-S)), len(np.where(Ss == 0)[0])
    else:
        Ss = S.copy()
        Ss[n:] = 0.0

    return np.dot(U, np.dot(np.diag(Ss), V))

def make_random_noise(x, rate, amplitude=5.0):
    noise = np.random.random(x.shape)
    noise[noise < (1.0-rate)] = 0
    noise[noise >= (1.0-rate)] = amplitude
    return x + noise


def generate_toy_data(M, N, toy_rank, toy_card=0.2):
	lr = np.random.rand(N, toy_rank)
	toy_card = 0.2

	ind = np.floor(np.random.rand(M)*toy_rank).astype(np.int)
	X0 = lr[:, ind]
	X0 = X0 - np.mean(X0, 0)


	S0 = np.sign(np.random.random((N, M))-0.5)
	S0[np.random.random((N, M)) >= toy_card] = 0.0
	X = X0 + S0

	return X, X0, S0

