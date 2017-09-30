
import tensorflow as tf
import numpy as np
from ts_pp import generate_ts

from util import *

from fista_model.model import *

seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

def make_random_noise(x, rate, amplitude=10.0):
    noise = np.random.random(x.shape)
    noise[noise < (1.0-rate)] = 0
    noise[noise >= (1.0-rate)] = amplitude
    return x + noise

sess = tf.Session()

seq_size = 2000
batch_size = 1
input_size = 1
filter_len = 50
layer_size = 100



D = np.random.randn(filter_len, 1, input_size, layer_size).astype(np.float32)

def gen_ts(seq_size):
    x = np.zeros((seq_size, batch_size, input_size))
    for bi in xrange(batch_size):
        for ni in xrange(input_size):
            x[:,bi,ni] = np.diff(generate_ts(seq_size+1))
            x[:,bi,ni] /= np.std(x[:,bi,ni])
    return x.transpose((1, 0, 2)).reshape((batch_size, seq_size, 1, input_size))

x = gen_ts(seq_size)
xt0 = np.concatenate((gen_ts(seq_size/2), gen_ts(seq_size/2)/5.0), 1)
xt1 = make_random_noise(gen_ts(seq_size), 0.005, 2.0)


m = SparseFistaModel(
    seq_size, batch_size, input_size, filter_len, layer_size, 
    alpha=0.01, step=1.0, lrate=1e-10, D=D*0.01
)

# m = LSModel(
#     seq_size, batch_size, input_size, filter_len, layer_size, 
#     lrate=1e-03, D=D*0.01
# )


tf.Session()
sess.run(tf.global_variables_initializer())

if isinstance(m, LSModel):

    ctx = m.init_ctx()

    ctx = LSModel.run_until_convergence(
        lambda c: m.run_dictionary_learning_step(sess, x),
        ctx=ctx,
        max_epoch=1000,
        tol=1e-06
    )

    x_hat, h_v, error_v = ctx.debug
    shl(x, x_hat)

    # xt_v = gen_ts()

    xt_v = np.concatenate((gen_ts(seq_size/2), gen_ts(seq_size/2)/5.0), 1)

    (re0, xh0), (re1, xh1) = (
        m.get_reconstruction_error(sess, x),
        m.get_reconstruction_error(sess, xt_v)
    )

    shl(re0, re1, show=False)
    shl(x, xh0, show=False)
    shl(xt_v, xh1, show=True)

elif isinstance(m, SparseModel):
    ctx = m.init_ctx()
    ctx = SparseModel.run_until_convergence(
        lambda c: m.run_encode_step(sess, x),
        ctx=ctx,
        max_epoch=1000, 
        tol=1e-06
    )
    
elif isinstance(m, SparseFistaModel):
    ctx = m.init_ctx()
    ctx = SparseModel.run_until_convergence(
        lambda c: m.run_encode_step(sess, x),
        ctx=ctx,
        max_epoch=1000, 
        tol=1e-06
    )
    shm(ctx.code)

    # ht0 = m.encode(sess, xt0)
    # ht1 = m.encode(sess, xt1)

    # (re, xh), (re0, xh0), (re1, xh1) = (
    #     m.get_reconstruction_error(sess, x, ctx.code),
    #     m.get_reconstruction_error(sess, xt0, ht0),
    #     m.get_reconstruction_error(sess, xt1, ht1)
    # )

    # shl(xt0, xh0, show=True)

