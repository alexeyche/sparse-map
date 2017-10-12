
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





class LCALayer(object):
    def __init__(self, batch_size, filter_len, input_size, layer_size):
        (
            self.batch_size, 
            self.input_size, 
            self.filter_len,
            self.layer_size
        ) = (
            batch_size, 
            input_size, 
            filter_len,
            layer_size
        )

        self.F = 1.0 * (np.random.uniform(size=(filter_len * input_size, layer_size)) - 0.5)
        self.F = normf(self.F)
        self.F_init = self.F.copy()

        self.Fc = np.dot(self.F.T, self.F) - np.eye(self.layer_size)
        self.Fc_init = self.Fc.copy()

        self.a_m = np.zeros((batch_size, layer_size))

        self.init_config()

    def init_config(self, **kwargs):
        c = Config()

        c.weight_init_factor = 0.3
        c.epsilon = 1.0
        c.tau = 5.0
        c.grad_accum_rate = 1.0

        # c.lam = 1.0
        c.lam = 0.01
        c.adaptive_threshold = False

        c.tau_m = 1000.0
        c.adapt = 1.0
        c.act_factor = 1.0
        c.adaptive = False

        c.tau_fb = 10.0
        c.fb_factor = 2.0
        c.smooth_feedback = False

        c.lrate = 0.05
        c.opt = SGDOpt((1.0, 1.0))

        c.update(kwargs)
        c.opt.init(self.F, self.Fc)

        self.c = c
        return c

        
    def init(self, seq_size):
        self.x_hat = np.zeros((seq_size, self.batch_size, self.input_size))

        self.u = np.zeros((self.batch_size, self.layer_size))
        self.a = np.zeros((self.batch_size, self.layer_size))
        self.dF = np.zeros(self.F.shape)
        self.dFc = np.zeros(self.Fc.shape)

        self.a_seq = np.zeros((seq_size, self.batch_size, self.layer_size))
        self.u_seq = np.zeros((seq_size, self.batch_size, self.layer_size))
        self.a_m_seq = np.zeros((seq_size, self.batch_size, self.layer_size))

        self.x_win = np.zeros((self.batch_size, self.filter_len, self.input_size))
        self.err_acc = 0.0


    def __call__(self, x, ti):
        left_ti = max(0, ti-filter_len)

        self.x_win[:, (filter_len-ti+left_ti):filter_len, :] = np.transpose(x[left_ti:ti], (1, 0, 2))
        
        x_flat = self.x_win.reshape(self.batch_size, self.filter_len * self.input_size)

        if self.c.adaptive_threshold:
            threshold = 10.0*np.mean(self.a_m)
        else:
            threshold = self.c.lam

        new_du = (- self.u + np.dot(x_flat, self.F) - np.dot(self.a, self.Fc))/ self.c.tau

        self.u += self.c.epsilon * new_du

        self.a[:] = act(self.u - threshold)

        self.a_m += self.c.epsilon * (self.c.adapt * self.a - self.a_m)/self.c.tau_m


        x_hat_flat_t = np.dot(self.a, self.F.T)
        
        error_part = x_flat - x_hat_flat_t
        
        self.err_acc += np.linalg.norm(error_part)
        
        self.dF += np.dot(error_part.T, self.a)
        self.dFc += np.dot(self.a.T, self.a)

        x_hat_t = x_hat_flat_t.reshape((self.batch_size, self.filter_len, self.input_size))
        self.x_hat[left_ti:ti] += np.transpose(x_hat_t[:, :(ti-left_ti), :], (1, 0, 2))/self.filter_len

        self.a_seq[ti] = self.a
        self.u_seq[ti] = self.u
        self.a_m_seq[ti] = self.a_m


    def learn(self):
        if np.linalg.norm(self.dFc) > 1000.0:
            raise Exception(str(np.linalg.norm(self.dFc)))

        self.F, self.Fc = self.c.opt.update((self.F, -self.dF), (self.Fc, -self.dFc))
        
        self.F = normf(self.F)
        
        # Fc = normf(Fc)
        # self.Fc = np.minimum(self.Fc, 0.5)
        self.Fc = np.dot(self.F.T, self.F) - np.eye(self.layer_size)
        

epochs = 10

np.random.seed(5)

input_size = 1
seq_size = 1000
batch_size = 1
layer_size = 100
filter_len = 25


act = relu

x = np.zeros((seq_size, batch_size, input_size))

for bi in xrange(batch_size):
    for ni in xrange(input_size):
        # x[:,bi,ni] = generate_ts(seq_size)
        x[:,bi,ni] = np.diff(generate_ts(seq_size+1))

# x[:,0,0] = 1.0*np.sin(np.linspace(0, 250, seq_size)/10.0)

layer = LCALayer(batch_size, filter_len, input_size, layer_size)

layer.init_config(
    adaptive_threshold=True,
    lam=0.005,
    opt=SGDOpt((1.0, 1.0)),
    # opt=MomentumOpt((0.05, 0.05), 0.9)
    # opt=AdamOpt((0.005, 0.005), beta1=0.9, beta2=0.999, eps=1e-05),
)


# opt = AdamOpt((0.01, 0.01), beta1=0.9, beta2=0.999, eps=1e-05)
# opt = NesterovMomentumOpt((0.1, 0.1), 0.99)
# opt = SGDOpt((5.0, 5.0))


a_hist = []
try:
    for e in xrange(1000):
        layer.init(seq_size)

        for ti in xrange(seq_size):
            layer(x, ti)

        layer.learn()

        error_profile = np.sum(np.square(layer.x_hat-x), 2)
        error = np.sum(error_profile)
        
        if e % 10 == 0:
            a_hist.append(layer.a_seq.copy())
        print "Epoch {}, err_acc {:.4f}, MSE {:.8f}".format(
            e, 
            layer.err_acc,
            error,
        )
except KeyboardInterrupt:
    pass

a_hist = np.asarray(a_hist)

# shm(a_seq, show=False)
# shl(a_m_seq)
shl(x, layer.x_hat, show=True)
# shm(F-F_init)
