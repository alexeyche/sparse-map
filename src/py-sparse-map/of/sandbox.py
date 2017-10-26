import tensorflow as tf

input_size, layer_size = 5, 2

D = tf.Variable(np.ones((input_size, layer_size)), dtype=tf.float32)

x = tf.placeholder(tf.float32, shape=(input_size, ))
y = tf.placeholder(tf.float32, shape=(output_size, ))

u = tf.matmul(x, D)

y_hat = tf.nn.softmax(u)

r = y - y_hat

dD = tf.matmul(x, r)


tf.gradients(tf.nn.l2_loss(y_hat - y))

x_v = np.random.random(input_size)
y_v = np.asarray([1.0, 0.0])

sess.run((u, y_hat, r, dD)
