
import tensorflow as tf
import numpy as np

class SparseLayer(object):
	def __init__(self, batch_size, input_size, layer_size, tau, h, adapt_gain, tau_m, feedback_const, ):
		self.batch_size = batch_size
		self.input_size = input_size
		self.layer_size = layer_size

		init_range = 2.0 * np.sqrt(6. / (input_size + layer_size))
		self.D_init = init_range * np.random.uniform(size=(input_size, layer_size)) - init_range/2.0

		self.D = tf.Variable(self.D_init, dtype=tf.float32)

		self.a_m = tf.placeholder(tf.float32, shape=(layer_size, ))

		self.h = h
		self.adapt_gain = adapt_gain
		self.tau_m = tau_m
		self.feedback_const = feedback_const
		self.tau = tau
	@property
	def init_state(self):
		return tf.zeros(dtype=tf.float32, shape=(self.batch_size, self.layer_size)), tf.zeros(dtype=tf.float32, shape=(self.batch_size, self.layer_size))

	def __call__(self, state, I):
		u, a = state

		I_hat = tf.matmul(a, tf.transpose(self.D))

		r = I - I_hat
		
		se = tf.reduce_sum(tf.square(r))
		
		gain = tf.matmul(r, self.D)

		new_u = u + self.h * (-u + gain)/self.tau

		return (new_u, tf.nn.relu(new_u - self.a_m)), se, r

	def final_state(self, state):
		u, a = state

		return u, a, (self.adapt_gain*tf.reduce_mean(a, 0) - self.a_m)/self.tau_m

	def feedback(self, state, top_down_signal):
		u, a = state

		return u + self.feedback_const * top_down_signal, a



class AELayer(object):
	def __init__(self, batch_size, input_size, layer_size):
		self.batch_size = batch_size
		self.input_size = input_size
		self.layer_size = layer_size

		init_range = 2.0 * np.sqrt(6. / (input_size + layer_size))
		self.D_init = init_range * np.random.uniform(size=(input_size, layer_size)) - init_range/2.0

		self.D = tf.Variable(self.D_init, dtype=tf.float32)

		self.act = tf.nn.relu

	def __call__(self, I):
		a = self.act(tf.matmul(I, self.D))

		I_hat = tf.matmul(a, tf.transpose(self.D))

		r = I - I_hat
		
		se = tf.reduce_sum(tf.square(r))
		
		return (a, ), se