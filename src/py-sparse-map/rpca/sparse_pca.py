
import tensorflow as tf
import numpy as np

def exp_nonlin(v, t, p):
    return tf.assign(v, 1.0 - tf.exp(-t/p))

def l0_nonlin_tf(v, t, p):
    k = int(t.get_shape()[0].value*(1.0-p))
    
    top_k_rev = tf.nn.top_k(-tf.abs(t), k=k)
    
    return tf.scatter_update(v, top_k_rev.indices, tf.zeros(top_k_rev.indices.get_shape()))



def find_components_tf(n, x, p, max_iter):
    C = np.cov(x.T)
    
    batch_size, input_size = x.shape

    A = tf.placeholder(tf.float32, shape=(input_size, input_size), name="A")
    r = tf.Variable(np.random.randn(input_size), dtype=tf.float32)

    new_r = tf.matmul(tf.expand_dims(r, 0), A)

    new_r = tf.squeeze(new_r)

    new_r = l0_nonlin_tf(r, new_r, p)  # 1/4

    new_r = tf.nn.l2_normalize(new_r, 0)
    new_r = tf.assign(r, new_r)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    def find_pc(pi, x):
        sess.run(tf.global_variables_initializer())

        for e in xrange(max_iter):
            rv = sess.run(new_r, {A: np.cov(x.T)})
            if e % 5 == 0:
                print "{}, iteration {}".format(pi, e)
        return rv

    last_pca, current_x = None, x
    pcs = []
    for pi in xrange(n):
        if not last_pca is None:
            current_x = current_x - np.outer(last_pca, np.dot(x, last_pca)).T
        pc = find_pc(pi, current_x)
        pcs.append(pc)

        last_pca = pc
    return np.asarray(pcs)


def l0_nonlin(t, p):
    k = int(t.shape[0]*(1.0-p))
    t[np.argsort(np.abs(t))[:k]] = 0.0
    return t

def find_components(n, x, p, max_iter):
    batch_size, input_size = x.shape

    def find_pc(pi, x):
        A = np.cov(x.T)
        r = np.random.randn(input_size)
        for e in xrange(max_iter):
            r = np.dot(r, A)
            if p < 1.0:
                r = l0_nonlin(r, p)
            r = r/np.linalg.norm(r, 2)
        return r

    last_pca, current_x = None, x
    pcs = []
    for pi in xrange(n):
        if not last_pca is None:
            current_x = current_x - np.outer(last_pca, np.dot(x, last_pca)).T
        pc = find_pc(pi, current_x)
        pcs.append(pc)

        last_pca = pc
    return np.asarray(pcs)



if __name__ == '__main__':
    from util import *
    from sklearn.datasets import load_iris

    np.random.seed(10)

    data = load_iris()
    x_v = data.data

    pcs = find_components(2, x_v, 0.25, 20)
    PC_est = np.dot(x_v, pcs.T)
    shs(
        PC_est[np.where(data.target == 0)],
        PC_est[np.where(data.target == 1)],
        PC_est[np.where(data.target == 2)],
        labels=["red", "blue", "green"],
    )