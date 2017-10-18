
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import platform
from util import *

x_size = 28
y_size = 28
xf_size = 8
yf_size = 8
layer_size = 144
batch_size = 100
lam = 0.1
threshold = 1e-03
h = 0.001
lrate = 1e-02

np.random.seed(42)

S = lambda x: tf.log(1.0 + tf.square(x))
dS = lambda x: 2.0 * x / (tf.square(x) + 1.0)

##


init_range = 2.0 * np.sqrt(6. / (xf_size + yf_size + layer_size))
D_init = init_range * np.random.uniform(size=(xf_size, yf_size, 1, layer_size)) - init_range/2.0

# D_init = D_init/np.linalg.norm(
# 	np.linalg.norm(
# 		np.linalg.norm(
# 			D_init, axis=0, keepdims=True
# 		), axis=1, keepdims=True
# 	), axis=2, keepdims=True
# )

D = tf.Variable(D_init, dtype=tf.float32)

##

I = tf.placeholder(tf.float32, shape=(batch_size, x_size, y_size, 1), name="I")
u_input = tf.placeholder(tf.float32, shape=(batch_size, x_size, y_size, layer_size), name="u")
a_input = tf.placeholder(tf.float32, shape=(batch_size, x_size, y_size, layer_size), name="a")

se_arr = []

u, a = u_input, a_input
for _ in xrange(10):
	I_hat = tf.nn.conv2d_transpose(
		a,
		D,
		I.get_shape(),
		strides=[1, 1, 1, 1], 
		padding='SAME',
		name="I_hat"
	)

	r = I - I_hat
	
	se = tf.reduce_sum(tf.square(r))
	# se = tf.Print(se, [se])

	gain = tf.nn.conv2d(
	    r,
	    D, 
	    strides=[1, 1, 1, 1], 
	    padding='SAME', 
	    name="gain"
	)

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
	
	for e in xrange(3):
		for bi in xrange(num_batches):
			a_v = np.zeros(a.get_shape().as_list())
			u_v = np.zeros(u.get_shape().as_list())

			x_v, y_v = mnist.train.next_batch(batch_size) 

			I_v = x_v.reshape((batch_size, x_size, y_size, 1))

			a_v, u_v, se_v, _ = sess.run(
				(
					a, 
					u, 
					se, 
					# se_arr, 
					# lam * dS(a),
					apply_grad_step
				), 
				{
					I: I_v,
					a_input: a_v,
					u_input: u_v
				}
			)
			# break
			print "Epoch {}, b {}, SE {:.4f}".format(e, bi, se_v)
		# break
except KeyboardInterrupt:
	pass