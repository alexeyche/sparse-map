import tensorflow as tf
import numpy as np
from util import *
import os
from of.dataset import get_toy_data, one_hot_encode
from of.model import ClassificationLayer, ClassificationLayerNonRec
from of.model import bounded_relu

input_size = 20

x_v, target_v = get_toy_data(input_size, 2000)
y_v = one_hot_encode(target_v)

output_size = y_v.shape[1]

# layer_size = 144
layer_size = 200
batch_size = 2000
lam = 0.1
lrate = 0.05

np.random.seed(42)
tf.set_random_seed(42)

#################

config = dict(
	tau = 5.0,
	tau_m = 1000.0,
	adapt_gain = 1.0,
	h = 0.1,
)


x = tf.placeholder(tf.float32, shape=(None, input_size), name="x")
y = tf.placeholder(tf.float32, shape=(None, output_size), name="y")

batch_size = tf.shape(x)[0]

net_structure = (output_size,)
layers_num = len(net_structure)

init_states = tuple(
	(
		tf.placeholder(tf.float32, shape=(None, ls), name="mem"),
		tf.placeholder(tf.float32, shape=(None, ls), name="y")
	) 
	for ls in net_structure
)



net = []

for l_id, lsize in enumerate(net_structure[:-1]):
	input_to_layer = input_size if l_id == 0 else net_structure[l_id-1]


	net.append(
		SparseLayer(batch_size, input_to_layer, lsize, **config)
	)


net.append(
	ClassificationLayer(
		batch_size, 
		net_structure[-2] if len(net_structure) > 1 else input_size, 
		output_size,
		act = bounded_relu,
		**config
	)
)


states = list(init_states)
# states = [l.init_state for l in net]
residuals = [None]*layers_num

states_acc = []
residuals_acc = []

for _ in xrange(30):
	errors = []	
		
	for l_id, l in enumerate(net):
		input_to_layer = x if l_id == 0 else states[l_id-1][-1]

		states[l_id], residuals[l_id] = l(states[l_id], input_to_layer, y)
	
	# input_to_last_layer = x if len(net) == 1 else states[-2][-1]
	# states[-1], residuals[-1] = net[-1](states[-1], input_to_last_layer, y)
	to_propagate = residuals[-1]
	
	for l_id, l in reversed(list(enumerate(net))):
		if l_id == len(net)-1:
			continue
		states[l_id] = l.feedback(states[l_id], to_propagate)	


	error = tf.nn.l2_loss(states[-1][-1] - y)
	errors.append(error)

	# states[0] = (tf.Print(states[0][0], [error]),) + states[0][1:]

	states_acc.append([tf.identity(ss) for s in states for ss in s])
	residuals_acc.append([tf.identity(rr) for rr in residuals])

final_states = tuple([l.final_state(ls) for l, ls in zip(net, states)])


# optimizer = tf.train.AdamOptimizer(lrate)
optimizer = tf.train.GradientDescentOptimizer(lrate)

grads_and_vars = [
	g_v 
	for l_id, (l, s) in enumerate(zip(net, states))
	   for g_v in l.get_grads_and_vars(s, x if l_id == 0 else states[l_id-1][-1])
]

apply_grads_step = tf.group(
    optimizer.apply_gradients(grads_and_vars),
)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size_v = x_v.shape[0]

y_m_v = [np.zeros(l.y_m.get_shape().as_list()) for l in net]

init_state_fn = lambda: tuple(
	tuple(
		np.zeros([batch_size_v, ] + s.get_shape().as_list()[1:])
	 	for s in init_state
	)
	for init_state in init_states
)

init_states_v = init_state_fn()


labels = ["green" if np.where(vv)[0] == 0 else "red" for vv in y_v]

def test(epoch):
	feeds = {x: x_v}
	feeds.update(
		dict([(l.y_m, y_m_v_l) for l, y_m_v_l in zip(net, y_m_v)])
	)
	fs_v, Dv = sess.run(
		( 
			final_states, 
			tuple([l.D for l in net])
		), feeds
	)
	shs(fs_v[-1][1], labels=(labels,), file="{}/tmp/toy_{}.png".format(os.environ["HOME"], epoch), figsize=(25,10))

try:

	ccc = None
	for e in xrange(1):
		feeds = {
			x: x_v, 
			y: y_v,
			init_states: init_states_v
		}

		feeds.update(
			dict([(l.y_m, y_m_v_l) for l, y_m_v_l in zip(net, y_m_v)])
		)
		sess_res = sess.run(
			(
				final_states,
				errors, 
				states_acc,
				residuals_acc,
				grads_and_vars,
			) + (
				(apply_grads_step,) if e > 0 else ()
			), 
			feeds
		)
		final_state_v, se_v = sess_res[:2]
		
		init_states_v = tuple(
			tuple(t.copy() for t in s[:2])
			for s in final_state_v
		)

		a_m_v = tuple(fsv[-1] for fsv in final_state_v)

		se_acc = np.asarray(se_v)
		if np.linalg.norm(se_acc) < 1e-10:
			raise KeyboardInterrupt
		
		if not ccc is None and se_acc[0] - ccc > 2.0:
			print "To big error: {}".format(se_acc[0] - ccc)
			raise KeyboardInterrupt

		ccc = se_acc[0]

		# if e % 5 == 0:
		# 	test(e)
	
		print "Epoch {}, SE {}".format(e, ", ".join(["{:.4f}".format(se_l) for se_l in se_acc]))
		# break


except KeyboardInterrupt:
	pass

s_acc = np.squeeze(np.asarray(sess_res[2]))
r_acc = np.squeeze(np.asarray(sess_res[3]))
# test("final")
shl(s_acc[:,1,0,:])