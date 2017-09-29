

import tensorflow as tf
import numpy as np
from util import *

from ts_pp import generate_ts

np.random.seed(5)
tf.set_random_seed(5)


batch_size = 1
input_size = 1
seq_size = 2000
filter_len = 50
layer_size = 100
lrate = 1e-02

# shrink = lambda a, b: tf.nn.relu(tf.abs(a) - b) * tf.sign(a)
shrink = lambda a, b: tf.nn.relu(tf.abs(a) - b)

####

x = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, input_size), name="x")

D_init = np.random.randn(filter_len, input_size, layer_size)*0.1
# D_init = generate_dct_dictionary(filter_len, layer_size).reshape((filter_len, input_size, layer_size))

D = tf.Variable(D_init.reshape((filter_len, 1, input_size, layer_size)), dtype=tf.float32)

h = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, layer_size), name="h")

x_hat = tf.nn.conv2d_transpose(
	h,
	D,
	x.get_shape(),
	strides=[1, 1, 1, 1], 
	padding='SAME',
	name="x_hat"
)

error = x - x_hat

loss = tf.reduce_mean(tf.square(error))

D_grad = tf.gradients(loss, [D])[0]

##

# optimizer = tf.train.AdadeltaOptimizer(lrate)
optimizer = tf.train.AdamOptimizer(lrate)
# optimizer = tf.train.GradientDescentOptimizer(lrate)

apply_grads_step = tf.group(
    optimizer.apply_gradients([(D_grad, D)]),
    tf.nn.l2_normalize(D, 0)
)

##

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_v = np.zeros((seq_size, batch_size, input_size))
for bi in xrange(batch_size):
    for ni in xrange(input_size):
        x_v[:,bi,ni] = np.diff(generate_ts(seq_size+1))
        x_v[:,bi,ni] /= np.std(x_v[:,bi,ni])
        # x_v[:,bi,ni] = generate_ts(seq_size)

        # x_v[:,bi,ni] = np.random.randn(seq_size)

x_v = x_v.transpose((1, 0, 2)).reshape((batch_size, seq_size, input_size, 1))


h_v = np.random.random(h.get_shape().as_list())
h_v[h_v < 0.999] = 0

e_m_arr, l_m_arr = [], []

lookback, tol = 10, 1e-05

try:
	for e in xrange(1000):

		x_hat_v, h_v, loss_v, D_v, error_v, _ = sess.run(
			[
				x_hat,
				h,
				loss,
				D,
				error,
				apply_grads_step
			],
			{
				x: x_v,
				h: h_v,
			}
		)
		e_m, l_m = np.mean(loss_v), np.mean(x_v)
		
		e_m_arr.append(e_m)
		l_m_arr.append(l_m)

		if e>lookback and np.std(e_m_arr[-lookback:]) < tol and np.std(l_m_arr[-lookback:]) < tol:
			print "Converged"
			break
		
		print "Epoch {}, loss {}, |h| {}".format(e, e_m, l_m)
except KeyboardInterrupt:
	pass
