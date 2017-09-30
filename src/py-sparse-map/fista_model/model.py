
import tensorflow as tf
import numpy as np
from collections import namedtuple

# shrink = lambda a, b: tf.nn.relu(tf.abs(a) - b) * tf.sign(a)
shrink = lambda a, b: tf.nn.relu(tf.abs(a) - b)


class SparseModelOld(object):

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
            
        self.filter_len = filter_len        
        
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

        self.se_error = tf.square(self.error)
        self.mse = tf.reduce_mean(self.se_error)

        self.D_grad = tf.gradients(self.mse, [self.D])[0]

        ##

        # optimizer = tf.train.AdadeltaOptimizer(lrate)
        optimizer = tf.train.AdamOptimizer(lrate)
        # optimizer = tf.train.GradientDescentOptimizer(lrate)

        self.apply_grads_step = tf.group(
            optimizer.apply_gradients([(self.D_grad, self.D)]),
            tf.nn.l2_normalize(self.D, 0)
        )

    def init_ctx(self):
        return SparseModel.Ctx(
            code=np.zeros(self.h.get_shape().as_list()), 
            sparsity=np.inf, 
            error=np.inf, 
            debug=()
        )


    def encode(self, session, data):
        ctx = self.init_ctx()
        ctx = SparseModel.run_until_convergence(
            lambda c: self.run_encode_step(session, data, c.code),
            ctx=ctx,
            max_epoch=1000, 
            tol=1e-06
        )
        return ctx.code

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

        return SparseModel.Ctx(code=h, error=mse, sparsity=np.mean(h), debug=(x_hat,error))


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

        return SparseModel.Ctx(code=code, error=mse, sparsity=np.mean(code), debug=(x_hat, error))


    def get_reconstruction_error(self, session, data, code):
        se_error, x_hat = session.run(
            (self.se_error, self.x_hat),
            {
                self.x: data,
                self.h: code
            }
        )

        return se_error[:,self.filter_len:-self.filter_len], x_hat

class LSModel(object):
    Ctx = namedtuple("Ctx", ["error", "debug"])

    
    @staticmethod
    def run_until_convergence(cb, ctx, max_epoch, tol, lookback=10):
        e_m_arr = []
        
        try:
            for epoch in xrange(max_epoch):

                ctx = cb(ctx)
                
                e_m_arr.append(ctx.error)
            
                if epoch>lookback and np.std(e_m_arr[-lookback:]) < tol:
                    print "Converged"
                    break
                
                print "Epoch {}, loss {}".format(epoch, ctx.error)
        except KeyboardInterrupt:
            pass

        return ctx


    def __init__(self, seq_size, batch_size, input_size, filter_len, layer_size, lrate, D=None):
        self.filter_len = filter_len
        
        self.x = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, input_size), name="x")
        
        D_init = np.random.randn(filter_len, 1, input_size, layer_size)
        
        self.D = tf.Variable(D_init if D is None else D, dtype=tf.float32)
        
        self.h = tf.nn.relu(
            tf.nn.conv2d(
                self.x, 
                self.D, 
                strides=[1, 1, 1, 1], 
                padding='SAME', 
                name="h"
            )
        )

        self.x_hat = tf.nn.conv2d_transpose(
            self.h,
            self.D,
            self.x.get_shape(),
            strides=[1, 1, 1, 1], 
            padding='SAME',
            name="x_hat"
        )

        self.error = self.x - self.x_hat
        self.se_error = tf.square(self.error)

        self.mse = tf.reduce_mean(self.se_error)

        self.D_grad = tf.gradients(self.mse, [self.D])[0]

        ##

        # optimizer = tf.train.AdadeltaOptimizer(lrate)
        optimizer = tf.train.AdamOptimizer(lrate)
        # optimizer = tf.train.GradientDescentOptimizer(lrate)

        self.apply_grads_step = tf.group(
            optimizer.apply_gradients([(self.D_grad, self.D)]),
            # tf.nn.l2_normalize(self.D, 0)
        )

    
    def init_ctx(self):
        return LSModel.Ctx(error=np.inf, debug=())


    def run_dictionary_learning_step(self, session, data):
        x_hat, code, mse, error, _ = session.run(
            [
                self.x_hat,
                self.h,
                self.mse,
                self.error,
                self.apply_grads_step,
            ],
            {
                self.x: data,
            }
        )

        return LSModel.Ctx(error=mse, debug=(x_hat, code, error))

    def get_reconstruction_error(self, session, data):
        se_error, x_hat = session.run(
            (self.se_error, self.x_hat),
            {
                self.x: data,
            }
        )

        return se_error[:,self.filter_len:-self.filter_len], x_hat


class SparseModel(object):

    Ctx = namedtuple("Ctx", ["code", "mse", "x_hat", "error", "sparsity"])

    @staticmethod
    def run_until_convergence(cb, ctx, max_epoch, tol, lookback=10):
        e_m_arr, l_m_arr = [], []
        
        try:
            for epoch in xrange(max_epoch):

                ctx = cb(ctx)
                
                e_m_arr.append(ctx.mse)
                l_m_arr.append(ctx.sparsity)

                if epoch>lookback and np.std(e_m_arr[-lookback:]) < tol and np.std(l_m_arr[-lookback:]) < tol:
                    print "Converged"
                    break
                
                print "Epoch {}, loss {}, |h| {}".format(epoch, ctx.mse, ctx.sparsity)
        except KeyboardInterrupt:
            pass

        return ctx


    def __init__(self, seq_size, batch_size, input_size, filter_len, layer_size, alpha, step, lrate, D=None):
        self.x = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, input_size), name="x")
        self.t = tf.placeholder(tf.float32, shape=(), name="t")
            
        self.filter_len = filter_len        
        
        D_init = np.random.randn(filter_len, 1, input_size, layer_size)
        
        self.D = tf.Variable(D_init if D is None else D, dtype=tf.float32)
        self.L = tf.square(tf.norm(D))

        self.alpha = alpha
        self.step = step
        
        ### IN

        self.h = tf.nn.conv2d(
            self.x, 
            self.D, 
            strides=[1, 1, 1, 1], 
            padding='SAME', 
            name="h"
        )
        
        for _ in xrange(5):
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
            
            self.h = shrink(self.h + self.step * self.h_grad/self.L, self.alpha/self.L)

        self.se_error = tf.square(self.error)
        self.mse = tf.reduce_mean(self.se_error)

        self.D_grad = tf.gradients(self.mse, [self.D])[0]

        ##

        # optimizer = tf.train.AdadeltaOptimizer(lrate)
        optimizer = tf.train.AdamOptimizer(lrate)
        # optimizer = tf.train.GradientDescentOptimizer(lrate)

        self.apply_grads_step = tf.group(
            optimizer.apply_gradients([(self.D_grad, self.D)]),
            tf.nn.l2_normalize(self.D, 0)
        )

    def init_ctx(self):
        return SparseModel.Ctx(
            code=None,
            x_hat=None,
            sparsity=np.inf, 
            mse=np.inf, 
            error=None,
        )


    def run_encode_step(self, session, data):
        x_hat, h, mse, error, _ = session.run(
            [
                self.x_hat,
                self.h,
                self.mse,
                self.error,
                self.apply_grads_step
            ],
            {
                self.x: data,
            }
        )

        return SparseModel.Ctx(code=h, x_hat=x_hat, mse=mse, error=error, sparsity=np.mean(h))


class SparseFistaModel(object):

    Ctx = namedtuple("Ctx", ["code", "mse", "x_hat", "error", "sparsity"])

    @staticmethod
    def run_until_convergence(cb, ctx, max_epoch, tol, lookback=10):
        e_m_arr, l_m_arr = [], []
        
        try:
            for epoch in xrange(max_epoch):

                ctx = cb(ctx)
                
                e_m_arr.append(ctx.mse)
                l_m_arr.append(ctx.sparsity)

                if epoch>lookback and np.std(e_m_arr[-lookback:]) < tol and np.std(l_m_arr[-lookback:]) < tol:
                    print "Converged"
                    break
                
                print "Epoch {}, loss {}, |h| {}".format(epoch, ctx.mse, ctx.sparsity)
        except KeyboardInterrupt:
            pass

        return ctx


    def __init__(self, seq_size, batch_size, input_size, filter_len, layer_size, alpha, step, lrate, D=None):
        self.x = tf.placeholder(tf.float32, shape=(batch_size, seq_size, 1, input_size), name="x")
        
        self.filter_len = filter_len        
        
        D_init = np.random.randn(filter_len, 1, input_size, layer_size)
        
        self.D = tf.Variable(D_init if D is None else D, dtype=tf.float32)
        self.L = tf.square(tf.norm(D))

        self.alpha = alpha
        self.step = step
        
        ### IN

        z = tf.nn.conv2d(
            self.x,
            self.D, 
            strides=[1, 1, 1, 1], 
            padding='SAME', 
            name="h"
        )
        
        # z = tf.zeros((batch_size, seq_size, 1, layer_size), dtype=tf.float32)
        t = tf.constant(1.0)
        
        h = z
        errors = []
        for _ in xrange(50):
            x_hat = tf.nn.conv2d_transpose(
                z,
                self.D,
                self.x.get_shape(),
                strides=[1, 1, 1, 1], 
                padding='SAME',
                name="x_hat"
            )

            error = self.x - x_hat

            # A.T.dot
            z_grad = tf.nn.conv2d(
                error, 
                self.D, 
                strides=[1, 1, 1, 1], 
                padding='SAME', 
                name="z_grad"
            )
            
            new_h = shrink(z + self.step * z_grad/self.L, self.alpha/self.L)
            new_t = (1.0 + tf.sqrt(1.0 + 4.0 * tf.square(t))) / 2.0
            
            z = new_h + ((t - 1.0) / new_t) * (new_h - h)
            h = new_h
            t = new_t

            # self.error = tf.Print(self.error, [tf.reduce_mean(tf.square(self.error))])
            
            errors.append(error)
        
        self.x_hat = x_hat
        self.h = h
        
        # self.error = tf.concat(errors,0)
        self.error = errors[-1]
        self.se_error = tf.square(self.error)
        self.mse = tf.reduce_mean(self.se_error)

        self.D_grad = tf.gradients(self.mse, [self.D])[0]

        ##

        # optimizer = tf.train.AdadeltaOptimizer(lrate)
        # optimizer = tf.train.AdamOptimizer(lrate)
        optimizer = tf.train.GradientDescentOptimizer(lrate)

        self.apply_grads_step = tf.group(
            optimizer.apply_gradients([(self.D_grad, self.D)]),
            # tf.nn.l2_normalize(self.D, 0)
        )

    def init_ctx(self):
        return SparseFistaModel.Ctx(
            code=None,
            x_hat=None,
            sparsity=np.inf, 
            mse=np.inf, 
            error=None
        )


    def run_encode_step(self, session, data):
        x_hat, h, mse, error, _ = session.run(
            [
                self.x_hat,
                self.h,
                self.mse,
                self.error,
                self.apply_grads_step
            ],
            {
                self.x: data
            }
        )

        return SparseFistaModel.Ctx(code=h, x_hat=x_hat, mse=mse, error=error, sparsity=np.mean(h))
