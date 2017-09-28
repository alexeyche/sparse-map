
import tensorflow as tf
import numpy as np
from collections import namedtuple

# shrink = lambda a, b: tf.nn.relu(tf.abs(a) - b) * tf.sign(a)
shrink = lambda a, b: tf.nn.relu(tf.abs(a) - b)


class Model(object):

	Ctx = namedtuple("Ctx", ["code", "error", "sparsity", "debug"])

	@staticmethod
	def run_until_convergence(cb, ctx, max_epoch, tol, lookback=10):
		e_m_arr, l_m_arr = [], []
		
		try:
			for epoch in xrange(max_epoch):

				ctx = cb(ctx)
				
				e_m_arr.append(ctx.error)
				l_m_arr.append(ctx.sparsity)

				if epoch>lookback and np.std(e_m_arr[-lookback:]) < tol and np.std(l_m_arr[-lookback:]) < tol:
					print "Converged"
					break
				
				print "Epoch {}, loss {}, |h| {}".format(epoch, ctx.error, ctx.sparsity)
		except KeyboardInterrupt:
			pass

		return ctx


	def __init__(self, seq_size, batch_size, input_size, filter_len, layer_size, alpha, step, lrate, D=None):
		self.x = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, input_size), name="x")
		self.t = tf.placeholder(tf.float32, shape=(), name="t")
		
		
		D_init = np.random.randn(filter_len, 1, input_size, layer_size)
		
		self.D = tf.Variable(D_init if D is None else D, dtype=tf.float32)
		self.L = tf.square(tf.norm(D))

		self.alpha = alpha
		self.step = step
		
		### IN
		
		self.h = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, layer_size), name="h")

		### OUT

		self.x_hat = tf.nn.conv2d_transpose(
			self.h,
			self.D,
			self.x.get_shape(),
			strides=[1, 1, 1, 1], 
			padding='SAME',
			name="x_hat"
		)

		self.error = self.x - self.x_hat

		# A.T.dot
		self.h_grad = tf.nn.conv2d(
			self.error, 
			self.D, 
			strides=[1, 1, 1, 1], 
			padding='SAME', 
			name="h_grad"
		)
		
		self.new_h = shrink(self.h + self.step * self.h_grad/self.L, self.alpha/self.L)

		self.mse = tf.reduce_mean(tf.square(self.error))

		self.D_grad = tf.gradients(self.mse, [self.D])[0]

		##

		# optimizer = tf.train.AdadeltaOptimizer(lrate)
		optimizer = tf.train.AdamOptimizer(lrate)
		# optimizer = tf.train.GradientDescentOptimizer(lrate)

		self.apply_grads_step = tf.group(
		    optimizer.apply_gradients([(self.D_grad, self.D)]),
		    tf.nn.l2_normalize(self.D, 0)
		)


	def run_encode_step(self, session, data, code):
		x_hat, h, mse, error = session.run(
			[
				self.x_hat,
				self.new_h,
				self.mse,
				self.error,
			],
			{
				self.x: data,
				self.h: code,
			}
		)

		return Model.Ctx(code=h, error=mse, sparsity=np.mean(h), debug=(x_hat,error))


	def run_dictionary_learning_step(self, session, data, code):
		x_hat, mse, error, _ = session.run(
			[
				self.x_hat,
				self.mse,
				self.error,
				self.apply_grads_step,
			],
			{
				self.x: data,
				self.h: code,
			}
		)

		return Model.Ctx(code=code, error=mse, sparsity=np.mean(code), debug=(x_hat, error))

