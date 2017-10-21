
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from util import *
import os
from of.model import AELayer

input_size = 28*28
layer_size = 144
batch_size = 100
lrate = 1e-04

np.random.seed(42)

rs = lambda x: x.reshape(28, 28)
##

	
################




I = tf.placeholder(tf.float32, shape=(None, input_size), name="I")

batch_size = tf.shape(I)[0]


layers_num = 3

l0, l1, l2 = (
	AELayer(batch_size, input_size, layer_size),
	AELayer(batch_size, layer_size, layer_size/2),
	AELayer(batch_size, layer_size/2, 2),
)

l0s, se0 = l0(I)
l1s, se1 = l1(l0s[-1])
l2s, se2 = l2(l1s[-1])


final_state0 = l0s
final_state1 = l1s
final_state2 = l2s



optimizer = tf.train.AdamOptimizer(lrate)
apply_grad_step = tf.group(
	optimizer.minimize(se0, var_list=[l0.D]),
	optimizer.minimize(se1, var_list=[l1.D]),
	optimizer.minimize(se2, var_list=[l2.D]),
)


mnist = input_data.read_data_sets(
	"{}/tmp/MNIST_data/".format(os.environ["HOME"]),
	one_hot=False
)



sess = tf.Session()
sess.run(tf.global_variables_initializer())


train_batch_size = 100
test_batch_size = 1000
num_batches = mnist.train.num_examples/train_batch_size

def test(epoch):
	x_v, y_v = mnist.test.next_batch(test_batch_size) 
	fs_v, Dv = sess.run(
		( 
			(final_state0, final_state1, final_state2), 
			(l0.D, l1.D, l2.D)
		), {
			I: x_v,
		}
	)
	shs(fs_v[-1][0], labels=(y_v,), file="{}/tmp/mnist_{}.png".format(os.environ["HOME"], epoch), figsize=(25,10))

try:
	
	for e in xrange(30):
		se_acc = np.zeros(layers_num)
		for bi in xrange(num_batches):
			x_v, y_v = mnist.train.next_batch(train_batch_size) 
			
			sess_res = sess.run(
				(
					(final_state0, final_state1, final_state2),
					(se0, se1, se2), 
					apply_grad_step
				), 
				{
					I: x_v
				}
			)
			final_state_v, se_v = sess_res[:2]
			
			se_acc += np.asarray(se_v)/num_batches
		
		if e % 5 == 0:
			test(e)

			
		print "Epoch {}, SE {}".format(e, ", ".join(["{:.4f}".format(se_l) for se_l in se_acc]))
		# break

	# 	a_v, Dv, I_hat_v = sess.run((a, D, I_hat), {I: x_v, layer.a_m: a_m_v})
except KeyboardInterrupt:
	pass

test("final")
