#!/usr/bin/env python3

import tensorflow as tf

n = 2
k = 3

F = tf.constant([-.1, 0, 0, -.1], tf.float64, [n, n], "system_matrix")
print(F)

x0 = tf.constant(1, tf.float64, [n, 1], "initial_state")
print(x0)

x = tf.matmul(F, x0, name="state")
print(x)

u = tf.constant([1, 2, 3, 4, 5, 6], tf.float64, [n, k], "input")

# i = tf.constant([0, 0], tf.int32, [2])
# xdims = tf.constant([n, 1], tf.int32, [2])

i = tf.constant(0, tf.int32)


c = lambda x, i: tf.less(i, k)

def b(x, i):
    x = tf.matmul(F, x)
    i = i + 1
    return x, i


loop = tf.while_loop(c, b, (x, i))

# u1 = tf.slice(u, i, xdims)

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(loop)[0])
    print(sess.run(u))
    # print(sess.run(u1))
