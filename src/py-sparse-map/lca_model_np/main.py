
from util import *

from config import Config
import numpy as np

from ts_pp import white_ts, generate_ts

from lca_model_np.opt import *

def relu(x):
    return np.maximum(x, 0.0)

def normf(F, axis=0):
    return F/np.linalg.norm(F, axis=axis)

def norm(w):
    return np.asarray([w[:,i]/np.sqrt(np.sum(np.square(w[:,i]))) for i in xrange(w.shape[1])]).T


epochs = 10

np.random.seed(5)

input_size = 1
seq_size = 250
batch_size = 1
layer_size = 50
filter_len = 25

c = Config()

c.weight_init_factor = 0.3
c.epsilon = 1.0
c.tau = 5.0
c.grad_accum_rate = 1.0

# c.lam = 1.0
c.lam = 0.02
c.adaptive_threshold = False

c.tau_m = 100.0
c.adapt = 1.0
c.act_factor = 1.0
c.adaptive = False

c.tau_fb = 10.0
c.fb_factor = 2.0
c.smooth_feedback = False

c.lrate = 0.05


act = relu

x = np.zeros((seq_size, batch_size, input_size))

for bi in xrange(batch_size):
    for ni in xrange(input_size):
        # x[:,bi,ni] = generate_ts(seq_size)
        x[:,bi,ni] = np.diff(generate_ts(seq_size+1))

# x[:,0,0] = 1.0*np.sin(np.linspace(0, 250, seq_size)/10.0)



F = 1.0 * (np.random.uniform(size=(filter_len * input_size, layer_size)) - 0.5)
F = normf(F)
F_init = F.copy()

Fc = np.dot(F.T, F) - np.eye(layer_size)
Fc_init = Fc.copy()

a_m = np.zeros((batch_size, layer_size))

# opt = AdamOpt((0.01, 0.01), beta1=0.9, beta2=0.999, eps=1e-05)
opt = NesterovMomentumOpt((0.5, 0.5), 0.99)
# opt = SGDOpt((20.5, 20.5))
opt.init(F, Fc)

try:
    for e in xrange(300):
        x_hat = np.zeros((seq_size, batch_size, input_size))

        u = np.zeros((batch_size, layer_size))
        a = np.zeros((batch_size, layer_size))
        dF = np.zeros(F.shape)
        dFc = np.zeros(Fc.shape)

        a_seq = np.zeros((seq_size, batch_size, layer_size))
        u_seq = np.zeros((seq_size, batch_size, layer_size))
        a_m_seq = np.zeros((seq_size, batch_size, layer_size))
        gain_seq = np.zeros((seq_size, batch_size, layer_size))
        feed_seq = np.zeros((seq_size, batch_size, layer_size))

        x_win = np.zeros((batch_size, filter_len, input_size))
        err_acc = 0.0
        for ti in xrange(seq_size):
            left_ti = max(0, ti-filter_len)

            x_win[:, (filter_len-ti+left_ti):filter_len, :] = np.transpose(x[left_ti:ti], (1, 0, 2))
            
            x_flat = x_win.reshape(batch_size, filter_len * input_size)

            if c.adaptive_threshold:
                threshold = a_m
            else:
                threshold = c.lam

            gain = np.dot(x_flat, F)
            feed = np.dot(a, Fc) 

            du = - u + gain - feed
            u += c.epsilon * du / c.tau

            a[:] = act(u - threshold)

            a_m += c.epsilon * (c.adapt * a - a_m)/c.tau_m


            x_hat_flat_t = np.dot(a, F.T)
            
            error_part = x_flat - x_hat_flat_t
            
            err_acc += np.linalg.norm(error_part)/seq_size
            
            dF += (1.0/seq_size) * np.dot(error_part.T, a)
            dFc += (1.0/seq_size) * np.dot(a.T, a)

            x_hat_t = x_hat_flat_t.reshape((batch_size, filter_len, input_size))
            x_hat[left_ti:ti] += np.transpose(x_hat_t[:, :(ti-left_ti), :], (1, 0, 2))/filter_len

            a_seq[ti] = a
            u_seq[ti] = u
            a_m_seq[ti] = a_m
            gain_seq[ti] = gain
            feed_seq[ti] = feed

        error_profile = np.mean(np.square(x_hat-x), 2)
        error = np.mean(error_profile)
        
        if np.linalg.norm(dF) > 1000.0:
            raise Exception(str(np.linalg.norm(dFc)))

        # F, Fc = opt.update((F, -dF), (Fc, -dFc))
        F, _ = opt.update((F, -dF), (Fc, -dFc))

        # F += c.lrate * dF
        # F = normf(F)
        
        # Fc += c.lrate * dFc
        # Fc = normf(Fc)
        Fc = np.dot(F.T, F) - np.eye(layer_size)

        print "Epoch {}, err_acc {:.4f}, MSE {:.8f}, |gain|l2: {:.4f}".format(
            e, 
            err_acc,
            error,
            np.mean(np.linalg.norm(gain_seq, axis=0))
        )
except KeyboardInterrupt:
    pass

# shm(a_seq, show=False)
# shl(a_m_seq)
shl(x, x_hat, show=True)
# shm(F-F_init)
