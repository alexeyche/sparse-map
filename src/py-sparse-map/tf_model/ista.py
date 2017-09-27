

import tensorflow as tf
import numpy as np
from util import *

from tf_model.ts_pp import generate_ts


np.random.seed(3)
tf.set_random_seed(3)


batch_size = 1
input_size = 1
seq_size = 400
filter_len = 50
layer_size = 100
alpha = 1.0
step = 1.0

# shrink = lambda a, b: tf.nn.relu(tf.abs(a) - b) * tf.sign(a)
shrink = lambda a, b: tf.nn.relu(tf.abs(a) - b)

####


x = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, input_size), name="x")

t = tf.placeholder(tf.float32, shape=(), name="t")

D_init = np.random.randn(filter_len, input_size, layer_size)*4.0
# D_init = generate_dct_dictionary(filter_len, layer_size).reshape((filter_len, input_size, layer_size))*0.1

D = tf.Variable(D_init.reshape((filter_len, 1, input_size, layer_size)), dtype=tf.float32)

L = tf.square(tf.norm(D))

h = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, layer_size), name="h")

# A.dot(z)
x_hat = tf.nn.conv2d_transpose(
	h,
	D,
	x.get_shape(),
	strides=[1, 1, 1, 1], 
	padding='SAME',
	name="x_hat"
)

error = x - x_hat

# A.T.dot
h_grad = tf.nn.conv2d(
	error, 
	D, 
	strides=[1, 1, 1, 1], 
	padding='SAME', 
	name="h_grad"
)

new_h = shrink(h + step * h_grad/L, alpha/L)

se = tf.reduce_sum(tf.square(error))

###

sess = tf.Session()
sess.run(tf.global_variables_initializer())

h_v = np.zeros(h.get_shape().as_list())

x_v = np.zeros((seq_size, batch_size, input_size))
for bi in xrange(batch_size):
    for ni in xrange(input_size):
        x_v[:,bi,ni] = generate_ts(seq_size)

x_v = x_v.transpose((1, 0, 2)).reshape((batch_size, seq_size, input_size, 1))

# x0 = np.random.random((seq_size, layer_size)).astype(np.float32)
# x0[x0 < 0.99] = 0
# x0 = x0.reshape((batch_size, seq_size, 1, layer_size))
# x_v = sess.run(tf.nn.conv2d_transpose(
# 	x0,
# 	D,
# 	x.get_shape(), 
# 	strides=[1, 1, 1, 1], 
# 	padding='SAME',
# 	name="b_hat"
# ))

e_m_arr, l_m_arr = [], []

tol = 1e-04
try:
	for e in xrange(2000):

		x_hat_v, h_v, L_v, h_grad_v, se_v, D_v = sess.run(
			[
				x_hat,
				new_h,
				L,
				h_grad, 
				se,
				D
			],
			{
				x: x_v,
				h: h_v,
			}
		)
		e_m, l_m = np.mean(se_v), np.mean(x_v)
		
		e_m_arr.append(e_m)
		l_m_arr.append(l_m)

		if e>10 and np.std(e_m_arr[-10:]) < tol and np.std(l_m_arr[-10:]) < tol:
			print "Converged"
			break
		
		print "Epoch {}, loss {}, |h| {}".format(e, e_m, l_m)
except KeyboardInterrupt:
	pass

# shl(np.asarray(e_m_arr), show=False)
shl(h_v, show=False)
shl(x_v, x_hat_v)

