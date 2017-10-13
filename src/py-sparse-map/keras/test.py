
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
lam = 0.01
threshold = 0.05
h = 0.0001

S = lambda x: tf.log(1.0 + tf.square(x))

##

D = tf.Variable(np.random.randn(xf_size, yf_size, 1, layer_size), dtype=tf.float32)
Df = tf.reshape(D, (xf_size*yf_size, layer_size))
Drec = tf.matmul(tf.transpose(Df), Df)

##

I = tf.placeholder(tf.float32, shape=(batch_size, x_size, y_size, 1), name="I")
a = tf.placeholder(tf.float32, shape=(batch_size, x_size, y_size, layer_size), name="a")


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

gain = tf.nn.conv2d(
    r,
    D, 
    strides=[1, 1, 1, 1], 
    padding='SAME', 
    name="gain"
)

sa = S(a)
dS = tf.gradients(sa, [a])[0]

# da = gain - lam*dS
da = tf.nn.relu(gain - threshold)



mnist = input_data.read_data_sets(
	"/home/alexeyche/tmp"
	if platform.system() == "Linux" else 
	"/Users/aleksei/tmp/MNIST_data/",
	one_hot=False
)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

a_v = np.zeros(a.get_shape().as_list())


num_batches = mnist.train.num_examples/batch_size

# for bi in xrange(num_batches):
x_v, y_v = mnist.train.next_batch(batch_size) 

I_v = x_v.reshape((batch_size, x_size, y_size, 1))

for it in xrange(10):

	da_v, se_v, dS_v = sess.run((da, se, dS), {
		I: I_v,
		a: a_v
	})

	a_v +=  h * da_v 

	print "Iteration No {}, SE {:.4f}".format(it, se_v)

shm(a_v[0,:,:,11], I_v[0,:,:])




