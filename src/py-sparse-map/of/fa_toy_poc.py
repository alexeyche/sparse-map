import tensorflow as tf
import numpy as np
from util import *
import os
from of.model import SparseLayer
from of.dataset import get_toy_data, one_hot_encode


def xavier_init(shape):
	init_range = 2.0 * np.sqrt(6. / np.sum(shape))
	return init_range * np.random.uniform(size=shape) - init_range/2.0


class ClassificationLayer(object):
	def __init__(self, batch_size, input_size, layer_size, h, tau):
		self.batch_size = batch_size
		self.input_size = input_size
		self.layer_size = layer_size

		self.D_init = xavier_init((input_size, layer_size))
		self.D = tf.Variable(self.D_init, dtype=tf.float32)

		self.act = tf.nn.softmax
		self.act_deriv = tf.nn.sigmoid
		
		self.h = h
		self.h_fb = -h

		self.tau = tau

	def __call__(self, state, x):
		mem, y = state
		
		gain = tf.matmul(x, self.D)

		new_mem = mem + self.h * (-mem + gain)/self.tau
		
		y = self.act(new_mem)
		return (new_mem, y)

	def get_derivative(self, state, target):
		mem, y = state
		return 2.0*(y - target) #* self.act_deriv(y)

	def get_grads_and_vars(self, state, x):
		mem, y = state
		
		r = x - tf.matmul(y, tf.transpose(self.D))

		dD = tf.matmul(tf.transpose(r), y*self.act_deriv(y))
		return [(dD, self.D)]

	def feedback(self, state, top_down_signal):
		mem, y = state
		new_mem = mem + self.h_fb * top_down_signal
		return new_mem, self.act(new_mem)

	@property
	def init_state(self):
		return (
			tf.zeros(dtype=tf.float32, shape=(self.batch_size, self.layer_size)), 
			tf.zeros(dtype=tf.float32, shape=(self.batch_size, self.layer_size))
		)
	
	def final_state(self, state):
		u, a = state

		return u, a


input_size = 20

x_v, target_v = get_toy_data(input_size, 2000)
y_v = one_hot_encode(target_v)

output_size = y_v.shape[1]

# layer_size = 144
layer_size = 200
batch_size = 2000
lam = 0.1
lrate = 1e-01

np.random.seed(42)
tf.set_random_seed(42)

#################

config = dict(
	tau = 5.0,
	# tau_m = 1000.0,
	# adapt_gain = 1000.0,
	h = 0.1,
)


x = tf.placeholder(tf.float32, shape=(None, input_size), name="x")

batch_size = tf.shape(x)[0]

net_structure = (output_size,)
layers_num = len(net_structure)

net = []

for l_id, lsize in enumerate(net_structure[:-1]):
	input_to_layer = input_size if l_id == 0 else net_structure[l_id-1]


	net.append(
		SparseLayer(batch_size, input_to_layer, lsize, **config)
	)


net.append(
	ClassificationLayer(batch_size, net_structure[-2] if len(net_structure) > 1 else input_size, output_size, **config)
)



states = [l.init_state for l in net]

states_acc = []

for _ in xrange(30):
	feedback = []
	
	errors = []	
		
	for l_id, l in enumerate(net):
		input_to_layer = x if l_id == 0 else states[l_id-1][-1]

		states[l_id] = l(states[l_id], input_to_layer)

		
	deriv = net[-1].get_derivative(states[-1], y_v)
	
	
	
	

	for l_id, l in reversed(list(enumerate(net))):
		# if l_id == len(net)-1:
		# 	continue
		states[l_id] = l.feedback(states[l_id], deriv)	


	error = tf.nn.l2_loss(states[-1][-1] - y_v)
	errors.append(error)

	states[0] = (tf.Print(states[0][0], [error]),) + states[0][1:]

	states_acc.append([tf.identity(ss) for s in states for ss in s])

final_states = tuple([l.final_state(ls) for l, ls in zip(net, states)])


optimizer = tf.train.AdamOptimizer(lrate)



apply_grads_step = tf.group(
    optimizer.apply_gradients(
    	[
    		g_v 
    		for l_id, (l, s) in enumerate(zip(net, states))
    		   for g_v in l.get_grads_and_vars(s, x if l_id == 0 else states[l_id-1][-1])
		]
	),
)



sess = tf.Session()
sess.run(tf.global_variables_initializer())


# a_m_v = [np.zeros(l.a_m.get_shape().as_list()) for l in net]

labels = ["green" if np.where(vv)[0] == 0 else "red" for vv in y_v]

def test(epoch):
	feeds = {x: x_v}
	# feeds.update(
	# 	dict([(l.a_m, a_m_v_l) for l, a_m_v_l in zip(net, a_m_v)])
	# )
	fs_v, Dv = sess.run(
		( 
			final_states, 
			tuple([l.D for l in net])
		), feeds
	)
	shs(fs_v[-1][1], labels=(labels,), file="{}/tmp/toy_{}.png".format(os.environ["HOME"], epoch), figsize=(25,10))

try:
	for e in xrange(1):
		feeds = {x: x_v}
		# feeds.update(
		# 	dict([(l.a_m, a_m_v_l) for l, a_m_v_l in zip(net, a_m_v)])
		# )
		sess_res = sess.run(
			(
				final_states,
				errors, 
				states_acc
			) + (
				(apply_grad_step,) if e > 0 else ()
			), 
			feeds
		)
		final_state_v, se_v = sess_res[:2]
		
		a_m_v = tuple(fsv[-1] for fsv in final_state_v)

		se_acc = np.asarray(se_v)

		# if e % 5 == 0:
		# 	test(e)
	
		print "Epoch {}, SE {}".format(e, ", ".join(["{:.4f}".format(se_l) for se_l in se_acc]))
		# break


except KeyboardInterrupt:
	pass

hist = np.squeeze(np.asarray(sess_res[2]))
# test("final")
