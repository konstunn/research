#!/usr/bin/env python3

# import numpy as np
import tensorflow as tf

n = 2
k = 3

F = tf.constant([.1, 0, 0, .1], tf.float64, [n, n], "system_matrix")
C = tf.constant([1, 0, 0, 1], tf.float64, [2, 2], "control_matrix")

x = tf.constant(1, tf.float64, [n, 1], "state")

u = tf.constant(10, tf.float64, [n, k], "input")

# i = tf.constant([0, 0], tf.int32, [2])
# xdims = tf.constant([n, 1], tf.int32, [2])

i = tf.constant(0, tf.int32)
j = tf.constant([0, 0], tf.int32)


def c(x, i, k):
    return tf.less(i, k)


def b(x, i, k):
    xs = tf.slice(x, [0, i], [2, 1])
    xp = tf.matmul(F, xs)
    x = tf.concat([x, xp], 1)
    i = i + 1
    return x, i, k


loop = tf.while_loop(c, b, [x, i, k],
                     [tf.TensorShape([2, None]), i.get_shape(),
                      tf.TensorShape([])])

# us = tf.slice(u, i, xdims)

with tf.Session() as sess:
    # print(sess.run(x))
    print(sess.run(loop)[0])
    print(sess.run(u))
    # print(sess.run(us))
