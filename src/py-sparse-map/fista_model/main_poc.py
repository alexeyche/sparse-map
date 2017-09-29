
import tensorflow as tf
import numpy as np
from ts_pp import generate_ts

from util import *

from fista_model.model import Model

seed = 5
tf.set_random_seed(seed)
np.random.seed(seed)

def make_random_noise(x):
	noise = np.random.random(x.shape)
	noise[noise < 0.999] = 0
	noise[noise >= 0.999] = 10.0
	return x + noise

sess = tf.Session()

seq_size = 1000
batch_size = 1
input_size, filter_len, layer_size = 1, 50, 100




D = np.random.randn(filter_len, 1, input_size, layer_size).astype(np.float32)

x_v = np.zeros((seq_size, batch_size, input_size))
for bi in xrange(batch_size):
    for ni in xrange(input_size):
        x_v[:,bi,ni] = np.diff(generate_ts(seq_size+1))
        x_v[:,bi,ni] /= np.std(x_v[:,bi,ni])
        # x_v[:,bi,ni] = generate_ts(seq_size)

        # x_v[:,bi,ni] = np.random.randn(seq_size)

x_v = x_v.transpose((1, 0, 2)).reshape((batch_size, seq_size, 1, input_size))
x_v = x_v.copy()
x_v[0,50] = 10.0
x_v[0,150] = -10.0
x_v[0,550] = 10.0



m = Model(
	seq_size,
	batch_size,
	input_size, 
	filter_len, 
	layer_size, 
	alpha=0.1, 
	step=1.0, 
	lrate=1e-02,
	D=D
)

tf.Session()
sess.run(tf.global_variables_initializer())

h_v = np.zeros(m.h.get_shape().as_list())

ctx = Model.Ctx(code=h_v, error=np.inf, sparsity=np.inf, debug=())


ctx = Model.run_until_convergence(
	lambda c: m.run_encode_step(sess, x_v, c.code),
	ctx=ctx,
	max_epoch=1000, 
	tol=1e-06
)

# for _ in xrange(10):
# 	ctx = Model.run_until_convergence(
# 		lambda c: m.run_encode_step(sess, x_v, c.code),
# 		ctx=ctx,
# 		max_epoch=1000, 
# 		tol=1e-06
# 	)
	
# 	ctx = Model.run_until_convergence(
# 		lambda c: m.run_dictionary_learning_step(sess, x_v, c.code), 
# 		ctx=ctx,
# 		max_epoch=1000, 
# 		tol=1e-08
# 	)


# ctx = Model.Ctx(code=h_v, error=np.inf, sparsity=np.inf, debug=())

# ctx = Model.run_until_convergence(
# 	lambda c: m.run_encode_step(sess, x_v_d, c.code),
# 	ctx=ctx,
# 	max_epoch=1, 
# 	tol=1e-06
# )