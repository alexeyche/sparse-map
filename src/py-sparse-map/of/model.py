
import tensorflow as tf
import numpy as np

def xavier_init(shape):
	init_range = 2.0 * np.sqrt(6. / np.sum(shape))
	return init_range * np.random.uniform(size=shape) - init_range/2.0

def bounded_relu(x):
	return tf.minimum(tf.nn.relu(x), 1.0)


class PredictiveCodingLayer(object):
	def __init__(
		self, 
		batch_size, 
		input_size, 
		layer_size,
		act=tf.nn.relu,
		h=1.0, 
		h_fb=-1.0,
		tau=5.0, 
		adapt_gain=1.0, 
		tau_m=1000.0
	):
		self.batch_size = batch_size
		self.input_size = input_size
		self.layer_size = layer_size

		self.h = h
		self.h_fb = -h
		self.adapt_gain = adapt_gain
		self.tau = tau
		self.tau_m = tau_m

		self.act = act

		self.D_init = xavier_init((input_size, layer_size))
		self.D = tf.Variable(self.D_init, dtype=tf.float32)

		self.y_m = tf.placeholder(tf.float32, shape=(layer_size, ))

		
	def __call__(self):
		raise NotImplementedError

	def get_grads_and_vars(self, state, x):
		mem, y = state
		
		dD = -tf.matmul(tf.transpose(x), y)

		# r = x - tf.matmul(y - self.y_m, tf.transpose(self.D))
		# r = x - tf.matmul(y, tf.transpose(self.D))
		# dD = -tf.matmul(tf.transpose(r), y)
		
		return [(dD/tf.cast(self.batch_size, tf.float32), self.D)]

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
		mem, y = state
		return mem, y, (self.adapt_gain*tf.reduce_mean(y, 0) - self.y_m)/self.tau_m


class ClassificationLayerNonRec(PredictiveCodingLayer):
	def __init__(
		self, 
		batch_size, 
		input_size, 
		layer_size, 
		**kwargs
	):
		super(ClassificationLayerNonRec, self).__init__(
			batch_size, 
			input_size, 
			layer_size, 
			**kwargs
		)


	def __call__(self, state, x, y_t):
		new_mem = tf.matmul(x, self.D)
		y_hat = self.act(new_mem)
		
		residuals = y_t - y_hat
		
		return (new_mem, residuals), y_hat, residuals


def way0(self, y_t, y, x):
	x_t = tf.matmul(y_t, tf.transpose(self.D))
	x_hat = tf.matmul(y, tf.transpose(self.D))

	residuals = x_t - x_hat

	gain = tf.matmul(residuals, self.D)

	return residuals, gain

def way1(self, y_t, y, x):
	x_t = tf.matmul(y_t, tf.transpose(self.D))
	x_hat = tf.matmul(tf.matmul(x, self.D), tf.transpose(self.D))
	residuals = x_t - x_hat

	return residuals, residuals, y_hat

def way2(self, y_t, y, x):
	y_hat = tf.matmul(x, self.D)
	
	residuals = tf.matmul(y_t - y_hat, tf.transpose(self.D))
	
	gain = tf.matmul(residuals, self.D)
	return residuals, gain, y_hat

def way3(self, y_t, y, x):
	y_hat = self.act(tf.matmul(x, self.D))
	
	residuals = y_t - y_hat
	
	gain = residuals
	return residuals, gain, y_hat

class ClassificationLayer(PredictiveCodingLayer):
	def __call__(self, state, x, y_t):
		mem, y = state
		
		residuals, gain = way0(self, y_t, y, x)
		# residuals, gain = way1(self, y_t, y, x)
		# residuals, gain = way2(self, y_t, y, x)
		# residuals, gain = way3(self, y_t, y, x)


		new_mem = mem + self.h * (-mem + gain)/self.tau
		
		
		return (new_mem, new_mem), y_hat, residuals


class ReconstructionLayer(PredictiveCodingLayer):
	def __call__(self, state, x, y_t):
		mem, y = state

		x_hat = tf.matmul(y, tf.transpose(self.D))

		residuals = x - x_hat

		gain = tf.matmul(residuals, self.D)

		new_mem = mem + self.h * (-mem + gain)/self.tau
		
		y = self.act(new_mem)
		return (new_mem, y), residuals



