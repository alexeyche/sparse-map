

import tensorflow as tf
import numpy as np
from util import *

from tf_model.ts_pp import generate_ts


np.random.seed(1)
tf.set_random_seed(1)


batch_size = 1
input_size = 1
seq_size = 500
filter_len = 100
layer_size = 50
alpha = 1.0
lrate = 1e-02 # Adam


shrink = lambda a, b: tf.nn.relu(tf.abs(a) - b) * tf.sign(a)

b_v = np.zeros((seq_size, batch_size, input_size))


for bi in xrange(batch_size):
    for ni in xrange(input_size):
        b_v[:,bi,ni] = generate_ts(seq_size)

####


b = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, input_size), name="b")

t = tf.placeholder(tf.float32, shape=(), name="t")

D = tf.get_variable("D", [filter_len, 1, input_size, layer_size], 
    initializer=tf.random_normal_initializer()
)

L = tf.reduce_sum(tf.square(D))

x = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, layer_size), name="x")
z = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, layer_size), name="z")

b_hat = tf.nn.conv2d(
	z,
	tf.transpose(D, (0, 1, 3, 2)), 
	strides=[1, 1, 1, 1], 
	padding='SAME',
	name="b_hat"
)

z_grad = tf.nn.conv2d(
	b - b_hat, 
	D, 
	strides=[1, 1, 1, 1], 
	padding='SAME', 
	name="z_grad"
)

new_x = shrink(z + z_grad/L, alpha/L)   # x

new_t = (1.0 + tf.sqrt(1.0 + 4.0 * t ** 2)) / 2.0

new_z = new_x + ((t - 1.0) / new_t) * (new_x - x)


b_hat_ans = tf.nn.conv2d(
	x,
	tf.transpose(D, (0, 1, 3, 2)), 
	strides=[1, 1, 1, 1], 
	padding='SAME',
	name="b_hat_ans"
)

error = tf.reduce_sum(tf.square(b - b_hat_ans))


###

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_v = np.zeros(x.get_shape().as_list())
z_v = x_v.copy()

t_v = 1
try:
	for e in xrange(100):

		b_hat_v, b_hat_ans_v, x_v, z_v, t_v, L_v, z_grad_v, error_v, D_v = sess.run(
			[
				b_hat,
				b_hat_ans,
				new_x,
				new_z,
				new_t,
				L,
				z_grad, 
				error,
				D
			],
			{
				b: b_v.transpose((1, 0, 2)).reshape((batch_size, seq_size, input_size, 1)),
				x: x_v,
				z: z_v,
				t: t_v,
			}
		)

		print "Epoch {}, loss {}, |h| {}".format(e, np.mean(error_v), np.mean(x_v))
except KeyboardInterrupt:
	pass


