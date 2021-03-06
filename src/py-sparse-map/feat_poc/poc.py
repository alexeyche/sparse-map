from util import *
from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
from matplotlib import pyplot as plt
import platform
from sklearn.datasets import make_classification

slice_by_target = lambda x, target: [x[np.where(target == t)] for t in np.unique(target)]



def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

class PCLayer(object):
	def __init__(self, x, layer_size, act, weight_sd):
		self.batch_size, self.input_size = x.get_shape().as_list()
		
		self.act = act
		self.layer_size = layer_size

		Wv = -weight_sd/2 + weight_sd*np.random.random((self.input_size, self.layer_size))
		# Wv = weight_sd*np.random.randn(self.input_size, self.layer_size)
		# Wv = Wv/np.linalg.norm(Wv, axis=0)

		self.W = tf.Variable(Wv, dtype=tf.float32)
		self.b = tf.Variable(np.zeros((layer_size,)), dtype=tf.float32)

		self.a = act(self.b + tf.matmul(x, self.W))

		self.x_hat = tf.matmul(self.a, tf.transpose(self.W))

		self.err = self.x_hat - x
		self.mse = tf.reduce_mean(tf.square(self.err))

		
	def get_local_grads(self):
		# dW = tf.matmul(tf.transpose(self.err), self.a)
		dW, db = tf.gradients(self.mse, [self.W, self.b])
		return [(dW, self.W), (db, self.b)]
		# return [(dW, self.W)]
		

mnist = input_data.read_data_sets(
	"/home/alexeyche/tmp"
	if platform.system() == "Linux" else 
	"/Users/aleksei/tmp/MNIST_data/",
	one_hot=False
)


np.random.seed(10)
tf.set_random_seed(10)


## iris

mnist = None

# data = load_iris()
# y_v = data.target
# x_v = data.data

x_v, y_v = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=2,
    n_redundant=4,
    n_repeated=0,
    n_clusters_per_class=2,
    n_classes=2,
    random_state=10
)



num_batches = 1

batch_size = x_v.shape[0]
input_size = x_v.shape[1]


# batch_size = 500
# input_size = mnist.train.images.shape[1]

local_rule = True
net_structure = [500, 250, 2]
lrates = [1.0] * len(net_structure)
# lrates = [1.0, 0.1, 0.001]

# act = tf.identity
# act = tf.nn.relu
# act = tf.nn.elu
# act = tf.nn.tanh
# act = tf.nn.sigmoid
act = lambda x: tf.sign(x) * tf.nn.relu(tf.abs(x))

x = tf.placeholder(tf.float32, shape=(batch_size, input_size), name="x")

net, inp = [], x
for layer_size in net_structure:
	net.append(
		PCLayer(inp, layer_size, act, weight_sd=0.5)
	)
	inp = net[-1].a


if local_rule:
	o = tf.train.AdamOptimizer(1e-03)
	# o = tf.train.AdadeltaOptimizer(1e-01)
	# o = tf.train.GradientDescentOptimizer(1e-06)
	# o = tf.train.AdagradOptimizer(1e-02)
	# o = tf.train.RMSPropOptimizer(1e-03)

	mse = [l.mse for l in net]

	apply_grads_step = tf.group(
		o.apply_gradients([
			(dW*lrate, W) 
			for (dW, W), lrate in zip(
				[g for l in net for g in l.get_local_grads()], 
				lrates
			)
		])
	)
else:
	o = tf.train.AdamOptimizer(1e-04)
	# o = tf.train.AdadeltaOptimizer(1e-01)

	rev_input = net[-1].a
	for l in reversed(net):
		rev_input = tf.matmul(rev_input, tf.transpose(l.W))
	
	mse = [tf.reduce_mean(tf.square(x - rev_input))]

	apply_grads_step = o.minimize(mse[0])


sess = tf.Session()
sess.run(tf.global_variables_initializer())
if not mnist is None:
	assert mnist.train.num_examples % batch_size == 0
	num_batches = mnist.train.num_examples/batch_size

try:
	for e in xrange(500):
		
		start_time = time.time()
		mse_v = np.zeros(len(mse))
		for bi in xrange(num_batches):
			if not mnist is None:
				x_v, y_v = mnist.train.next_batch(batch_size) 

			sess_vals = sess.run(
				[mse] + [(l.a, l.x_hat) for l in net] + [apply_grads_step]
				, 
				{
					x: x_v,
				}
			)
			
			mse_v += np.asarray(sess_vals[0])/num_batches
			net_vals = sess_vals[1:(1+len(net_structure))]

			a_v = [v[0] for v in net_vals]
			x_hat_v = [v[1] for v in net_vals]
			
		print "Epoch {} ({:.3f}s), MSE {}".format(
			e, 
			time.time()-start_time, 
			", ".join(["{:.3f}".format(vv) for vv in mse_v])
		)

except KeyboardInterrupt:
	pass

def get_pc(x_v):
	C = np.cov(x_v.T)
	eigval, eigvec = np.linalg.eig(C)
	return np.dot(x_v, eigvec)


PC = get_pc(x_v)

cmap = get_cmap(len(np.unique(y_v))+1)

if not mnist is None:
	pic_size = np.sqrt(x_v.shape[1]).astype(np.int32)
	ix_v = x_v.reshape((x_v.shape[0], pic_size, pic_size))

shs(*slice_by_target(PC[:, :2], y_v), labels=[cmap(v) for v in np.unique(y_v)], show=False)
shs(*slice_by_target(a_v[-1], y_v), labels=[cmap(v) for v in np.unique(y_v)], show=True)