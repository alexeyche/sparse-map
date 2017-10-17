
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from util import *
import platform
from matplotlib import pyplot as plt
from scipy import ndimage as ndi

np.random.seed(5)
tf.set_random_seed(5)



batch_size, input_size = 200, 28*28
layer_size = 49
weight_init_factor = 1.0
lam = 0.57
lrate = 1e-03


g = lambda x: tf.log(tf.cosh(x))
# g = lambda x: tf.abs(x)
# g = lambda x: - tf.exp( - tf.square(x)/2.0)

# g = tf.identity
# act = tf.nn.relu
act = tf.identity
# act = tf.nn.tanh
# act = tf.nn.elu



x = tf.placeholder(tf.float32, shape=(batch_size, input_size), name="x")

x_p = tf.extract_image_patches(tf.reshape(x, (batch_size, 28, 28, 1)), [1, 7, 7, 1], [1, 4, 4, 1], [1,1,1,1], 'VALID')

x_p_shape = x_p.get_shape().as_list()[1:]
input_size = x_p_shape[0]*x_p_shape[1]*x_p_shape[2]


x_p = tf.reshape(x_p, (batch_size, input_size))

W = tf.get_variable("W", (input_size, layer_size), 
	initializer=tf.uniform_unit_scaling_initializer(factor=weight_init_factor)
)


a = act(tf.matmul(x_p, W))

x_hat = tf.matmul(a, tf.transpose(W))

Lrec = tf.nn.l2_loss(x_hat - x_p) / batch_size
Lsparse = tf.reduce_sum(g(a)) / batch_size

L = lam * Lrec + Lsparse

optimizer = tf.train.AdamOptimizer(lrate)
# optimizer = tf.train.GradientDescentOptimizer(lrate)

if lam == 0.0:
	apply_grads_step = tf.group(
		optimizer.minimize(L),
		tf.assign(W, tf.nn.l2_normalize(W, 0))
	)
else:
	apply_grads_step = optimizer.minimize(L)





mnist = input_data.read_data_sets(
	"/home/alexeyche/tmp"
	if platform.system() == "Linux" else 
	"/Users/aleksei/tmp/MNIST_data/",
	one_hot=False
)
num_batches = mnist.train.num_examples/batch_size




sess = tf.Session()
sess.run(tf.global_variables_initializer())


for e in xrange(15):
	Lrec_acc, Lsparse_acc = 0.0, 0.0

	for b_id in xrange(num_batches):
		x_v, y_v = mnist.train.next_batch(batch_size) 

		x_p_v = sess.run(x_p, {x: x_v})


		a_v, Lrec_v, Lsparse_v, x_hat_v, W_v, _ = sess.run((a, Lrec, Lsparse, x_hat, W, apply_grads_step), {x: x_v})
		
		Lrec_acc += Lrec_v/num_batches
		Lsparse_acc += Lsparse_v/num_batches

	print "Epoch {}, Lrec {}, Lsparse {}".format(e, Lrec_acc, Lsparse_acc)



ncols = 7
for col_id in xrange(ncols):
	shm(*[W_v[:,f_id].reshape(x_p_shape)[:,:,10] for f_id in xrange(ncols*col_id, ncols*(col_id+1))], ncols=ncols, id=col_id)


