
import numpy as np
from config import Config

from lca_model_np.opt import *

def normf(F, axis=0):
    return F/np.linalg.norm(F, axis=axis)

def relu(x):
    return np.maximum(x, 0.0)

class LCALayer(object):
    def __init__(self, batch_size, filter_len, input_size, layer_size, act=relu):
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
        
        init_range = 2.0 * np.sqrt(6. / (filter_len * input_size + layer_size))
        
        self.act = act

        self.F = init_range * np.random.uniform(size=(filter_len * input_size, layer_size)) - init_range/2.0
        
        
        self.F = normf(self.F)
        self.F_init = self.F.copy()

        self.Fc = np.dot(self.F.T, self.F) - np.eye(self.layer_size)
        self.Fc_init = self.Fc.copy()

        self.a_m = np.zeros((batch_size, layer_size))

        self.init_config()

    def init_config(self, **kwargs):
        c = Config()

        c.weight_init_factor = 0.3
        c.epsilon = 0.1
        c.feedback_epsilon = -0.1
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
        self.td_seq = np.zeros((seq_size, self.batch_size, self.layer_size))
        self.a_m_seq = np.zeros((seq_size, self.batch_size, self.layer_size))

        self.x_win = np.zeros((self.batch_size, self.filter_len, self.input_size))
        self.err_acc = 0.0


    def __call__(self, x, ti):
        left_ti = max(0, ti-self.filter_len)
        
        self.x_win[:, (self.filter_len-ti+left_ti):self.filter_len, :] = np.transpose(x[left_ti:ti], (1, 0, 2))
        
        x_flat = self.x_win.reshape(self.batch_size, self.filter_len * self.input_size)

        if self.c.adaptive_threshold:
            threshold = self.a_m #2.0*np.mean(self.a_m)
        else:
            threshold = self.c.lam

        new_du = (- self.u + np.dot(x_flat, self.F) - np.dot(self.a, self.Fc))/ self.c.tau

        self.u += self.c.epsilon * new_du

        self.a[:] = self.act(self.u - threshold)

        self.a_m += self.c.epsilon * (self.c.adapt * self.a - self.a_m)/self.c.tau_m


        x_hat_flat_t = np.dot(self.a, self.F.T)
        
        residuals_flat = x_flat - x_hat_flat_t
        self.residuals = residuals_flat.reshape(self.batch_size, self.filter_len, self.input_size)
        
        self.err_acc += np.linalg.norm(residuals_flat)
        
        self.dF += np.dot(residuals_flat.T, self.a)
        self.dFc += np.dot(self.a.T, self.a)

        x_hat_t = x_hat_flat_t.reshape((self.batch_size, self.filter_len, self.input_size))
        self.x_hat[left_ti:ti] += np.transpose(x_hat_t[:, :(ti-left_ti), :], (1, 0, 2))/self.filter_len

        self.a_seq[ti] = self.a
        self.u_seq[ti] = self.u
        self.a_m_seq[ti] = self.a_m

    def feedback(self, top_down_signal, ti):
        self.u += self.c.feedback_epsilon * top_down_signal
        self.td_seq[ti] = top_down_signal.copy()


    def learn(self, sparse=False):
        if np.linalg.norm(self.dFc) > 1000.0:
            raise Exception(str(np.linalg.norm(self.dFc)))
        
        self.F, self.Fc = self.c.opt.update((self.F, -self.dF), (self.Fc, -self.dFc))

        if not sparse:
            self.F = normf(self.F)
            
            # Fc = normf(Fc)
            # self.Fc = np.minimum(self.Fc, 0.5)
            self.Fc = np.dot(self.F.T, self.F) - np.eye(self.layer_size)
        

