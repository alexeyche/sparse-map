
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from util import *
import platform
from matplotlib import pyplot as plt

np.random.seed(5)
tf.set_random_seed(5)



batch_size, input_size = 200, 28*28
layer_size = 100
weight_init_factor = 0.1
lam = 0.7
lrate = 0.00001


g = lambda x: tf.log(tf.cosh(x))

x = tf.placeholder(tf.float32, shape=(batch_size, input_size), name="x")


W = tf.get_variable("W", (input_size, layer_size), 
	initializer=tf.uniform_unit_scaling_initializer(factor=weight_init_factor)
)


a = tf.matmul(x, W)

x_hat = tf.matmul(a, tf.transpose(W))

Lrec = tf.nn.l2_loss(x_hat - x)
Lsparse = tf.reduce_sum(g(a))

L = lam * Lrec + Lsparse

optimizer = tf.train.AdamOptimizer(lrate)

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


for e in xrange(1):
	
	Lrec_acc, Lsparse_acc = 0.0, 0.0

	for b_id in xrange(num_batches):
		x_v, y_v = mnist.train.next_batch(batch_size) 
		
		a_v, Lrec_v, Lsparse_v, x_hat_v, W_v, _ = sess.run((a, Lrec, Lsparse, x_hat, W, apply_grads_step), {x: x_v})
		
		Lrec_acc += Lrec_v/float(num_batches)
		Lsparse_acc += Lsparse_v/float(num_batches)

	print "Epoch {}, Lrec {}, Lsparse {}".format(e, Lrec_acc, Lsparse_acc)




r = lambda x: x.reshape(28, 28)

# n=10; shm(r(x_hat_v[n]), r(x_v[n]), subplot_col=1)
# n=11; shm(x_hat_v[n].reshape(28, 28), x_v[n].reshape(28, 28), subplot=222)

# plt.show()

shm(r(W_v[:,0]), r(W_v[:,1]), subplot_col=2, plot_id=0, show=False)
shm(r(W_v[:,2]), r(W_v[:,3]), subplot_col=2, plot_id=1)





