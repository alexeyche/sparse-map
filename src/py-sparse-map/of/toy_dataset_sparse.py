import tensorflow as tf
import numpy as np
from util import *
import os
from of.model import SparseLayer
from sklearn.datasets import make_classification

def get_toy_data(dest_dim, size, n_classes=2, seed=2):
    x_values, y_values = make_classification(
        n_samples=size,
        n_features=2, 
        n_informative=2, 
        n_redundant=0, 
        n_repeated=0,
        n_clusters_per_class=2,
        n_classes=n_classes,
        scale=0.1,
        shift=5.0,
        random_state=seed
    )
    return quantize_data(x_values, dest_dim), y_values.astype(np.int32)

def quantize_data(x, dest_size):
    x_out = np.zeros((x.shape[0], dest_size))
    
    dim_size = x.shape[1]
    size_per_dim = dest_size/dim_size
    
    min_vals = np.min(x, 0)
    max_vals = np.max(x, 0)
    for xi in xrange(x.shape[0]):
        for di in xrange(dim_size):
            v01 = (x[xi, di] - min_vals[di]) / (max_vals[di] - min_vals[di])
            x_out[xi, int(di * size_per_dim + v01 * (size_per_dim-1))] = 1.0
    return x_out


input_size = 50

x_v, y_v = get_toy_data(input_size, 2000)

# layer_size = 144
layer_size = 200
batch_size = 2000
lam = 0.1
threshold = 0.1
lrate = 1e-01

np.random.seed(42)
tf.set_random_seed(42)

S = lambda x: tf.log(1.0 + tf.square(x))
dS = lambda x: 2.0 * x / (tf.square(x) + 1.0)

#################

config = dict(
	tau = 5.0,
	tau_m = 1000.0,
	adapt_gain = 1000.0,
	lam = 0.05,
	feedback_const = -0.05,
	h = 0.05,
)


I = tf.placeholder(tf.float32, shape=(None, input_size), name="I")

batch_size = tf.shape(I)[0]

net_structure = (layer_size, layer_size/2, 2)
layers_num = len(net_structure)

net = []

for l_id, lsize in enumerate(net_structure):
	input_to_layer = input_size if l_id == 0 else net_structure[l_id-1]
	if l_id == 1:
		config["lam"] = 0.0005
	if l_id == 2:
		config["lam"] = 1e-07

	net.append(
		SparseLayer(batch_size, input_to_layer, lsize, **config)
	)


states = [l.init_state for l in net]

for _ in xrange(30):
	feedback = []
	
	errors = [] 
		
	for l_id, l in enumerate(net):
		input_to_layer = I if l_id == 0 else states[l_id-1][-1]

		states[l_id], se, resid = l(states[l_id], input_to_layer)

		errors.append(se)
		feedback.append(resid)

	for l_id, l in reversed(list(enumerate(net))):
		if l_id == len(net)-1:
			continue
		l.feedback(states[l_id], feedback[l_id+1])	

final_states = tuple([l.final_state(ls) for l, ls in zip(net, states)])


optimizer = tf.train.AdamOptimizer(lrate)

apply_grad_step = tf.group(
	*[optimizer.minimize(se, var_list=[l.D])for se, l in zip(errors, net)]
)



sess = tf.Session()
sess.run(tf.global_variables_initializer())


a_m_v = [np.zeros(l.a_m.get_shape().as_list()) for l in net]

labels = ["green" if vv == 0 else "red" for vv in y_v]

def test(epoch):
	feeds = {I: x_v}
	feeds.update(
		dict([(l.a_m, a_m_v_l) for l, a_m_v_l in zip(net, a_m_v)])
	)
	fs_v, Dv = sess.run(
		( 
			final_states, 
			tuple([l.D for l in net])
		), feeds
	)
	shs(fs_v[-1][1], labels=(labels,), file="{}/tmp/toy_{}.png".format(os.environ["HOME"], epoch), figsize=(25,10))

try:
	for e in xrange(1000):
		feeds = {I: x_v}
		feeds.update(
			dict([(l.a_m, a_m_v_l) for l, a_m_v_l in zip(net, a_m_v)])
		)
		sess_res = sess.run(
			(
				final_states,
				errors, 
			) + (
				(apply_grad_step,) if e > 0 else ()
			), 
			feeds
		)
		final_state_v, se_v = sess_res[:2]
		
		a_m_v = tuple(fsv[-1] for fsv in final_state_v)

		se_acc = np.asarray(se_v)

		if e % 5 == 0:
			test(e)
	
		print "Epoch {}, SE {}".format(e, ", ".join(["{:.4f}".format(se_l) for se_l in se_acc]))
		# break

	# 	a_v, Dv, I_hat_v = sess.run((a, D, I_hat), {I: x_v, layer.a_m: a_m_v})
except KeyboardInterrupt:
	pass

test("final")
