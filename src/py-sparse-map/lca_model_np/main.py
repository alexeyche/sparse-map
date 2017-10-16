
from util import *

from config import Config
import numpy as np

from ts_pp import white_ts, generate_ts

from lca_model_np.opt import *
from lca_model_np.model import *


def norm(w):
    return np.asarray([w[:,i]/np.sqrt(np.sum(np.square(w[:,i]))) for i in xrange(w.shape[1])]).T





epochs = 10

np.random.seed(1)

input_size = 1
seq_size = 1000
batch_size = 1
layer_size = 100
filter_len = 24

act = relu

x = np.zeros((seq_size, batch_size, input_size))

for bi in xrange(batch_size):
    for ni in xrange(input_size):
        # x[:,bi,ni] = generate_ts(seq_size)
        x[:,bi,ni] = np.diff(generate_ts(seq_size+1))

# x[:,0,0] = 1.0*np.sin(np.linspace(0, 250, seq_size)/10.0)/50.0

net = [
    LCALayer(batch_size, filter_len, input_size, layer_size),
    LCALayer(batch_size, filter_len/2, layer_size, layer_size/2),
    LCALayer(batch_size, filter_len/4, layer_size/2, 10),
]

net[0].init_config(
    adaptive_threshold=True,
    lam=0.005,
    opt=SGDOpt((1.0, 1.0)),
    # opt=MomentumOpt((0.05, 0.05), 0.9)
    # opt=AdamOpt((0.005, 0.005), beta1=0.9, beta2=0.999, eps=1e-05),
)

net[1].init_config(
    adaptive_threshold=True,
    lam=0.005,
    opt=SGDOpt((2.0, 2.0)),
    # opt=MomentumOpt((0.05, 0.05), 0.9)
    # opt=AdamOpt((0.005, 0.005), beta1=0.9, beta2=0.999, eps=1e-05),
)

net[2].init_config(
    adaptive_threshold=True,
    lam=0.005,
    opt=SGDOpt((0.1, 0.1)),
    # opt=MomentumOpt((0.05, 0.05), 0.9)
    # opt=AdamOpt((0.005, 0.005), beta1=0.9, beta2=0.999, eps=1e-05),
)

# opt = AdamOpt((0.01, 0.01), beta1=0.9, beta2=0.999, eps=1e-05)
# opt = NesterovMomentumOpt((0.1, 0.1), 0.99)
# opt = SGDOpt((5.0, 5.0))


try:
    for e in xrange(500):
        [l.init(seq_size) for l in net]

        for ti in xrange(seq_size):
            layer_input = x
            for layer in net:
                layer(layer_input, ti)
                layer_input = layer.a_seq

        [l.learn() for l in net]
        
        layer_input = x
        error_profile = []
        for l in net:
            error_profile.append(np.sum(np.square(l.x_hat-layer_input), 2))
            layer_input = l.a_seq
        
        error = [np.sum(ep) for ep in error_profile]
        
        print "Epoch {}, err_acc {}, MSE {}".format(
            e, 
            ", ".join(["{:.4f}".format(l.err_acc) for l in net]),
            ", ".join(["{:.8f}".format(mse) for mse in error]),
        )
except KeyboardInterrupt:
    pass

# a_hist = np.asarray(a_hist)

# shm(a_seq, show=False)
# shl(a_m_seq)
# shl(x, net[0].x_hat, show=False)
# shm(net[0].a_seq, net[1].x_hat, show=False)
# shm(net[1].a_seq, net[2].x_hat, show=False)
shl(net[-1].a_seq)

# shm(F-F_init)
