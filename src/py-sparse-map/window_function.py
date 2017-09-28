import tensorflow as tf
import numpy as np



def map_over_seq(callback, *inputs, **kwargs):
    parallel_iterations = kwargs.get("parallel_iterations", 100)

    seq_size = inputs[0].get_shape()[0].value
    
    i0 = tf.constant(0)
    
    inputs_ta = tuple(
        tf.TensorArray(dtype=tf.float32, size=seq_size).unstack(input)
        for input in inputs
    )

    output_ta = tf.TensorArray(dtype=tf.float32, size=seq_size)

    def body(i, rt):
        return i+1, rt.write(i, callback(*[input.read(i) for input in inputs_ta]))


    _, out_arr = tf.while_loop(
        cond = lambda i, *_: tf.less(i, seq_size),
        body = body,
        loop_vars = (i0, output_ta),
        parallel_iterations = parallel_iterations
    )

    return out_arr.stack()


if __name__ == '__main__':
    import time

    seq_size, batch_size, input_size = 10000, 1, 10
    window = 10

    x = tf.placeholder(tf.float32, shape=(seq_size, batch_size, input_size), name="x")

    def callback(x):
        return tf.matmul(tf.transpose(x, (1, 0)), x)

    x_v = np.random.randn(*x.get_shape().as_list())

    sess = tf.Session()
    
    start_time = time.time()
    
    r_v = sess.run(
        map_over_seq(
            callback,
            x, 
            parallel_iterations = 1
        ), 
        {x: x_v}
    )

    print time.time() - start_time


    