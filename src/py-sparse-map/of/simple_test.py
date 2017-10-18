
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from util import *
import os

input_size = 28*28
layer_size = 144
batch_size = 100
lam = 0.1
threshold = 0.1
h = 0.1
lrate = 1e-03
tau_m = 1000.0
adapt_gain = 100.0

np.random.seed(42)

S = lambda x: tf.log(1.0 + tf.square(x))
dS = lambda x: 2.0 * x / (tf.square(x) + 1.0)
rs = lambda x: x.reshape(28, 28)
##

class Layer(object):
	def __init__(self, batch_size, input_size, layer_size):
		self.batch_size = batch_size
		self.input_size = input_size
		self.layer_size = layer_size

		init_range = 2.0 * np.sqrt(6. / (input_size + layer_size))
		self.D_init = init_range * np.random.uniform(size=(input_size, layer_size)) - init_range/2.0

		self.D = tf.Variable(self.D_init, dtype=tf.float32)

		self.a_m = tf.placeholder(tf.float32, shape=(layer_size, ))


	@property
	def init_state(self):
		return tf.zeros(dtype=tf.float32, shape=(self.batch_size, self.layer_size)), tf.zeros(dtype=tf.float32, shape=(self.batch_size, self.layer_size))

	def __call__(self, state, I):
		u, a = state

		I_hat = tf.matmul(a, tf.transpose(self.D))

		r = I - I_hat
		
		se = tf.reduce_sum(tf.square(r))
		
		gain = tf.matmul(r, self.D)

		new_u = u + h * (-u + gain)/5.0

		return (new_u, tf.nn.relu(new_u - self.a_m)), se

	def final_state(self, state):
		u, a = state

		return u, a, (adapt_gain*tf.reduce_mean(a, 0) - self.a_m)/tau_m

################




I = tf.placeholder(tf.float32, shape=(None, input_size), name="I")

batch_size = tf.shape(I)[0]

layers_num = 3

l0, l1, l2 = (
	Layer(batch_size, input_size, layer_size),
	Layer(batch_size, layer_size, layer_size/2),
	Layer(batch_size, layer_size/2, 2),
)

l0s = l0.init_state
l1s = l1.init_state
l2s = l2.init_state

for _ in xrange(10):	
	l0s, se0 = l0(l0s, I)
	l1s, se1 = l1(l1s, l0s[-1])
	l2s, se2 = l2(l2s, l1s[-1])


final_state0 = l0.final_state(l0s)
final_state1 = l1.final_state(l1s)
final_state2 = l2.final_state(l2s)




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


a_m_v = [np.zeros(l.a_m.get_shape().as_list()) for l in (l0, l1, l2)]

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
			l0.a_m: a_m_v[0], 
			l1.a_m: a_m_v[1],
			l2.a_m: a_m_v[2]
		}
	)
	shs(fs_v[-1][1], labels=(y_v,), file="{}/tmp/mnist_{}.png".format(os.environ["HOME"], epoch), figsize=(25,10))

try:
	
	for e in xrange(1):
		se_acc = np.zeros(layers_num)
		for bi in xrange(num_batches):
			x_v, y_v = mnist.train.next_batch(train_batch_size) 
			
			sess_res = sess.run(
				(
					(final_state0, final_state1, final_state2),
					(se0, se1, se2), 
				) + (
					(apply_grad_step,) if e > 0 else ()
				), 
				{
					I: x_v,
					l0.a_m: a_m_v[0],
					l1.a_m: a_m_v[1],
					l2.a_m: a_m_v[2]
				}
			)
			final_state_v, se_v = sess_res[:2]
			
			a_m_v = tuple(fsv[-1] for fsv in final_state_v)

			se_acc += np.asarray(se_v)/num_batches
		
		if e % 5 == 0:
			test(e)

			
		print "Epoch {}, SE {}".format(e, ", ".join(["{:.4f}".format(se_l) for se_l in se_acc]))
		# break

	# 	a_v, Dv, I_hat_v = sess.run((a, D, I_hat), {I: x_v, layer.a_m: a_m_v})
except KeyboardInterrupt:
	pass

test("final")
