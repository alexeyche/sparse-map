
from util import *

from config import Config
import numpy as np

from ts_pp import white_ts, generate_ts

def relu(x):
    return np.maximum(x, 0.0)

epochs = 10

np.random.seed(5)

input_size = 1
seq_size = 500
batch_size = 1
layer_size = 100
filter_len = 25

c = Config()

c.weight_init_factor = 0.3
c.epsilon = 1.0
c.tau = 5.0
c.grad_accum_rate = 1.0

c.lam = 0.0003
c.adaptive_threshold = False

c.tau_m = 50.0
c.adapt = 5.0
c.act_factor = 1.0
c.adaptive = False

c.tau_fb = 10.0
c.fb_factor = 2.0
c.smooth_feedback = False

c.lrate = 0.1


act = relu

x = np.zeros((seq_size, batch_size, input_size))

for bi in xrange(batch_size):
    for ni in xrange(input_size):
        x[:,bi,ni] = generate_ts(seq_size)
        x[:,bi,ni] = np.diff(generate_ts(seq_size+1))

x_hat = np.zeros((seq_size, batch_size, input_size))

F = np.random.randn(filter_len * input_size, layer_size)
F = F/np.linalg.norm(F, axis=0)
F_init = F.copy()

Fc = np.dot(F.T, F)
Fc_init = Fc.copy()

try:
    for e in xrange(100):    
        u = np.zeros((batch_size, layer_size))
        a = np.zeros((batch_size, layer_size))
        a_m = np.zeros((batch_size, layer_size))
        dF = np.zeros(F.shape)
        dFc = np.zeros(Fc.shape)

        a_seq = np.zeros((seq_size, batch_size, layer_size))
        u_seq = np.zeros((seq_size, batch_size, layer_size))
        a_m_seq = np.zeros((seq_size, batch_size, layer_size))

        x_win = np.zeros((filter_len, batch_size, input_size))

        for ti in xrange(seq_size):
            left_ti = max(0, ti-filter_len)

            x_win[(filter_len-ti+left_ti):filter_len] = x[left_ti:ti]

            x_flat = x_win.reshape(batch_size, filter_len * input_size)

            if c.adaptive_threshold:
                threshold = a_m
            else:
                threshold = c.lam

            du = - u + np.dot(x_flat, F) - np.dot(a, Fc) 
            u += c.epsilon * du / c.tau

            a[:] = act(u - threshold)


            a_m += c.epsilon * (c.adapt * a - a_m)/c.tau_m


            x_hat_flat_t = np.dot(a, F.T)
            
            error_part = x_hat_flat_t - x_flat

            dF += np.dot(error_part.T, a)
            dFc += - np.dot(a.T, a)

            x_hat_t = x_hat_flat_t.reshape((batch_size, filter_len, input_size))
            x_hat[left_ti:ti] += np.transpose(x_hat_t[:, :(ti-left_ti), :], (1, 0, 2))/20.0


            a_seq[ti] = a
            u_seq[ti] = u
            a_m_seq[ti] = a_m

        error_profile = np.mean(np.square(x_hat-x), 2)
        error = np.mean(error_profile)

        F += c.lrate * dF
        Fc += c.lrate * dFc
        
        print "Epoch {}, MSE {:}".format(
            e, 
            error
        )
except KeyboardInterrupt:
    pass


shl(x, x_hat, show=False)
shm(F-F_init)
