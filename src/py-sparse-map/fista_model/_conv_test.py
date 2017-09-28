
import tensorflow as tf
import numpy as np
from ts_pp import generate_ts

from util import *

batch_size = 1
input_size = 1
seq_size = 2000
layer_size = 100


def run(filter_len):
	np.random.seed(3)
	tf.set_random_seed(3)

	D_init = np.random.randn(filter_len, input_size, layer_size)
	# D_init = generate_dct_dictionary(filter_len, layer_size).reshape((filter_len, input_size, layer_size))*0.1

	D = tf.Variable(D_init.reshape((filter_len, 1, input_size, layer_size)), dtype=tf.float32)


	x = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, input_size), name="x")


	xc = tf.nn.conv2d(
		x, 
		D, 
		strides=[1, 1, 1, 1], 
		padding='VALID', 
		name="xc"
	)



	x_v = np.zeros((seq_size, batch_size, input_size))
	for bi in xrange(batch_size):
	    for ni in xrange(input_size):
	        x_v[:,bi,ni] = generate_ts(seq_size)
	x_v = x_v.transpose((1, 0, 2)).reshape((batch_size, seq_size, input_size, 1))

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	xc_v = sess.run(xc, {x: x_v})
	
	# dst_pic = "/home/alexeyche/tmp/xc_{0}_{1:.2f}.png".format(int(filter_len), factor)
	dst_pic = "/home/alexeyche/tmp/xc_{0}.png".format(int(filter_len))
	shl(xc_v, file=dst_pic)
	print filter_len

# for f in xrange(5, 1000, 25):
# 	run(f)

