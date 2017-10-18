
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import platform
from util import *

input_size = 28*28
layer_size = 144
batch_size = 100
lam = 0.1
threshold = 0.1
h = 0.1
lrate = 1e-04

np.random.seed(42)

S = lambda x: tf.log(1.0 + tf.square(x))
dS = lambda x: 2.0 * x / (tf.square(x) + 1.0)

##


init_range = 2.0 * np.sqrt(6. / (input_size + layer_size))
D_init = init_range * np.random.uniform(size=(input_size, layer_size)) - init_range/2.0

# D_init = D_init/np.linalg.norm(
# 	np.linalg.norm(
# 		np.linalg.norm(
# 			D_init, axis=0, keepdims=True
# 		), axis=1, keepdims=True
# 	), axis=2, keepdims=True
# )

D = tf.Variable(D_init, dtype=tf.float32)

##

I = tf.placeholder(tf.float32, shape=(batch_size, input_size), name="I")
u_input = tf.placeholder(tf.float32, shape=(batch_size, layer_size), name="u")
a_input = tf.placeholder(tf.float32, shape=(batch_size, layer_size), name="a")

se_arr = []

u, a = u_input, a_input
for _ in xrange(10):
	I_hat = tf.matmul(a, tf.transpose(D))

	r = I - I_hat
	
	se = tf.reduce_sum(tf.square(r))
	# se = tf.Print(se, [se])

	gain = tf.matmul(r, D)

	u += h * (-u + gain)/5.0

	a = tf.nn.relu(u - threshold)
	# a = tf.nn.relu(u - 10.0*tf.reduce_mean(a, axis=(0, 3), keep_dims=True))

	se_arr.append(se)


optimizer = tf.train.AdamOptimizer(lrate)
apply_grad_step = tf.group(
	optimizer.minimize(se),
	# tf.assign(D, tf.nn.l2_normalize(D, dim=(0,1,2)))
)


mnist = input_data.read_data_sets(
	"/home/alexeyche/tmp"
	if platform.system() == "Linux" else 
	"/Users/aleksei/tmp/MNIST_data/",
	one_hot=False
)



sess = tf.Session()
sess.run(tf.global_variables_initializer())



num_batches = mnist.train.num_examples/batch_size
try:
	
	for e in xrange(10):
		se_acc = 0.0
		for bi in xrange(num_batches):
			a_v = np.zeros(a.get_shape().as_list())
			u_v = np.zeros(u.get_shape().as_list())

			x_v, y_v = mnist.train.next_batch(batch_size) 

			a_v, u_v, se_v, se_arr_v, _ = sess.run(
				(
					a, 
					u, 
					se, 
					se_arr, 
					# lam * dS(a),
					apply_grad_step
				), 
				{
					I: x_v,
					a_input: a_v,
					u_input: u_v
				}
			)
			# break
			se_acc += se_v/num_batches

		print "Epoch {}, SE {:.4f}".format(e, se_acc)
		# break
	if e > 1:
		a_v = np.zeros(a.get_shape().as_list())
		u_v = np.zeros(u.get_shape().as_list())

		a_v, Dv, I_hat_v = sess.run((a, D, I_hat), {I: x_v, a_input: a_v, u_input: u_v})
except KeyboardInterrupt:
	pass