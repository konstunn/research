#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

n = 2
k = 10

th = tf.constant([.1, .1], )

x_shape = tf.TensorShape([n, 1])

F = tf.constant([.1, 0, 0, .1], tf.float64, [n, n], "system_matrix")
C = tf.constant([1, 0, 0, 1], tf.float64, [2, 2], "control_matrix")

x = tf.constant(1, tf.float64, x_shape, "state")

u = tf.constant(10, tf.float64, [n, k], "input")

i = tf.constant(0, tf.int32)


def cond(x, i):
    return tf.less(i, k)


def body(x, i):
    x_slice = tf.slice(x, [0, i], x_shape)
    xp = tf.matmul(F, x_slice)
    x = tf.concat([x, xp], 1)
    i = i + 1
    return x, i


loop = tf.while_loop(cond, body, [x, i],
                     [tf.TensorShape([2, None]), i.get_shape()])


with tf.Session() as sess:
    tf.summary.FileWriter('.tf', sess.graph)
    xx = sess.run(loop)[0]
    xx = np.transpose(xx)
    print(xx)
