

import tensorflow as tf
import numpy as np
from util import *

from tf_model.ts_pp import generate_ts


np.random.seed(1)
tf.set_random_seed(1)


batch_size = 1
input_size = 1
seq_size = 1000
filter_len = 100
layer_size = 50
weight_init_factor = 0.1
alpha = 0.5
lam = 0.01

x_v = np.zeros((seq_size, batch_size, input_size))

for bi in xrange(batch_size):
    for ni in xrange(input_size):
        x_v[:,bi,ni] = generate_ts(seq_size)

####


x = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, input_size), name="Input")

D = tf.get_variable("D", [filter_len, 1, input_size, layer_size], 
    initializer=tf.uniform_unit_scaling_initializer(factor=weight_init_factor)
)


h = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, layer_size), name="Input")

# h = tf.nn.conv2d(
# 	x, 
# 	D, 
# 	strides=[1, 1, 1, 1], 
# 	padding='SAME', 
# 	name="h"
# )

x_hat = tf.nn.conv2d(
	h,
	tf.transpose(D, (0, 1, 3, 2)), 
	strides=[1, 1, 1, 1], 
	padding='SAME',
	name="x_hat"
)

error = x_hat - x

h_grad = tf.nn.conv2d(
	error, 
	D, 
	strides=[1, 1, 1, 1], 
	padding='SAME', 
	name="h_grad"
)

new_h = h - alpha * h_grad

shrink = lambda a, b: tf.sign(a) * tf.maximum(tf.abs(a) - b, 0) 
new_h = shrink(new_h, alpha * lam)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# h_v = np.random.randn(*h.get_shape().as_list())
h_v = np.zeros(h.get_shape().as_list())

for e in xrange(50):

	x_hat_v, error_v, h_v, D_v, h_grad_v = sess.run(
		[x_hat, error, new_h, D, h_grad],
		{
			x: x_v.transpose((1, 0, 2)).reshape((batch_size, seq_size, input_size, 1)),
			h: h_v
		}
	)

	print "Epoch {}, loss {}, |h| {}".format(e, np.mean(error_v), np.mean(h_v))
