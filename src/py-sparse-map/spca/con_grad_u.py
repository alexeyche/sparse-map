

from util import *
from sklearn.datasets import load_iris
import tensorflow as tf

def exp_nonlin(v, t, p):
	return tf.assign(v, 1.0 - tf.exp(-t/p))

def l0_nonlin(v, t, p):
	k = int(t.get_shape()[0].value*(1.0-p))
	
	top_k_rev = tf.nn.top_k(-tf.abs(t), k=k)
	
	return tf.scatter_update(v, top_k_rev.indices, tf.zeros(top_k_rev.indices.get_shape()))


data = load_iris()
x_v = data.data

np.random.seed(10)
tf.set_random_seed(10)

batch_size, input_size = x_v.shape
dim_size = 2
p = 0.25


A = tf.placeholder(tf.float32, shape=(input_size, input_size), name="A")
r = tf.Variable(np.random.randn(input_size), dtype=tf.float32)

new_r = tf.matmul(tf.expand_dims(r, 0), A)

new_r = tf.squeeze(new_r)

new_r = l0_nonlin(r, new_r, p)

new_r = tf.nn.l2_normalize(new_r, 0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

Av = np.cov(x_v.T)

eigvec = np.linalg.eig(Av)[1]

PCA = np.dot(x_v, eigvec)[:,0:2]

def find_pc(pi, epochs, x_v):
	sess.run(tf.global_variables_initializer())

	Av = np.cov(x_v.T)

	for e in xrange(epochs):
		rv = sess.run(new_r, {A: Av})
		if e % 5 == 0:
			print "Epoch {}, {}".format(e, np.sum(np.abs(rv) - np.abs(eigvec[:,pi])))
	return rv



pc0 = find_pc(0, 20, x_v)
pc1 = find_pc(1, 20, x_v - np.outer(pc0, np.dot(x_v, pc0)).T)

PCA_est = np.dot(x_v, np.asarray([pc0, pc1]).T)

shs(
	PCA_est[np.where(data.target == 0)],
	PCA_est[np.where(data.target == 1)],
	PCA_est[np.where(data.target == 2)],
	labels=["red", "blue", "green"],
	show=False
)

shs(
	PCA[np.where(data.target == 0)],
	PCA[np.where(data.target == 1)],
	PCA[np.where(data.target == 2)],
	labels=["red", "blue", "green"],
)
