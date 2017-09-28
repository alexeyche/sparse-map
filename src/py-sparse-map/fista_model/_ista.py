

import tensorflow as tf
import numpy as np
from util import *

from ts_pp import generate_ts


np.random.seed(1)
tf.set_random_seed(1)


batch_size = 1
input_size = 1
seq_size = 500
filter_len = 100
layer_size = 50
weight_init_factor = 0.05
alpha = 0.2
lam = 1.0
lrate = 1e-02 # Adam
# lrate = 5e-02 # Adadelta
# lrate = 1e-04 # SGD

x_v = np.zeros((seq_size, batch_size, input_size))


for bi in xrange(batch_size):
    for ni in xrange(input_size):
        x_v[:,bi,ni] = generate_ts(seq_size)

####


x = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, input_size), name="Input")

D = tf.get_variable("D", [filter_len, 1, input_size, layer_size], 
    initializer=tf.uniform_unit_scaling_initializer(factor=weight_init_factor)
)


h_given = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, layer_size), name="Input")

h = tf.nn.conv2d(
	x, 
	D, 
	strides=[1, 1, 1, 1], 
	padding='SAME', 
	name="h"
)

for _ in xrange(3):
	x_hat = tf.nn.conv2d(
		h,
		tf.transpose(D, (0, 1, 3, 2)), 
		strides=[1, 1, 1, 1], 
		padding='SAME',
		name="x_hat"
	)

	error = x - x_hat


	h_grad = tf.nn.conv2d(
		error, 
		D, 
		strides=[1, 1, 1, 1], 
		padding='SAME', 
		name="h_grad"
	)



	# h = tf.nn.relu(tf.abs(h + alpha * h_grad) - lam * alpha) * tf.sign(h) 

l1Sparsity = tf.reduce_mean(tf.reduce_sum(tf.abs(h), reduction_indices=[1, 2, 3]))

rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(error), reduction_indices=[1, 2, 3]))
loss = rec_loss/2.0 + lam * l1Sparsity

D_grad = tf.gradients(loss, [D])[0]

##

# optimizer = tf.train.AdadeltaOptimizer(lrate)
optimizer = tf.train.AdamOptimizer(lrate)
# optimizer = tf.train.GradientDescentOptimizer(lrate)

apply_grads_step = tf.group(
    optimizer.apply_gradients([(D_grad, D)]),
    # tf.nn.l2_normalize(D, 0)
)



sess = tf.Session()
sess.run(tf.global_variables_initializer())
# D_before = sess.run(D)
# sess.run(tf.assign(D, tf.nn.l2_normalize(D, 0)))
# raise Exception()
# h_v = np.random.randn(*h.get_shape().as_list())
# h_v = np.zeros(h.get_shape().as_list())
try:
	for e in xrange(1000):

		x_hat_v, h_v, error_v, h_v, D_v, D_grad_v, h_grad_v, _ = sess.run(
			[
				x_hat,
				h, 
				error, 
				h, 
				D, 
				D_grad,
				h_grad,
				apply_grads_step,
			],
			{
				x: x_v.transpose((1, 0, 2)).reshape((batch_size, seq_size, input_size, 1)),
			}
		)

		print "Epoch {}, loss {}, |h| {}".format(e, np.mean(error_v), np.mean(h_v))
except KeyboardInterrupt:
	pass
