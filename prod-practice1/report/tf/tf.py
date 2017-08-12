#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

print('Defining graph')

n = 2
k = 10
p = 2
r = 2
m = 2

x_shape = tf.TensorShape([n, 1])

# TODO: these all should be python functions of theta, computed and results
# passed in placeholders
F = tf.constant([-.1, 0, 0, -.1], tf.float64, [n, n], "system_matrix")
C = tf.constant([1, 0, 0, 1], tf.float64, [r, n], "control_matrix")
G = tf.constant([1, 0, 0, 1], tf.float64, [n, p], "object_noise_matrix")
H = tf.constant([1, 0, 0, 1], tf.float64, [m, n], "measurement_matrix")
x0 = tf.constant([0, 0], tf.float64, x_shape, "initial_state_mean")
P0 = tf.constant([.1, .1], tf.float64, [n, n], "initial_state_covariance")

x = tf.constant([1, 1], tf.float64, x_shape, "state")

u = tf.constant(5, tf.float64, [r, k], "input")

i = tf.constant(0, tf.int32)


# TODO: use different distribution class instances for different vectors
mean = [10., 20, 30, 40, 50]
sigma = [1., 2, 3, 4, 5]
dist = tf.contrib.distributions.MultivariateNormalDiag(mean, sigma)
rv = dist.sample([5])


w_mean = [0., 0]

w_sigma = [[.1, 0],
           [0, .1]]

Q = tf.constant(w_sigma, tf.float32, [2, 2], "object_noise_covariance")

w_dist = tf.contrib.distributions.MultivariateNormalDiag(w_mean, Q)


def loop_cond(x, i, k):
    return tf.less(i, k)


def loop_body(x, i, k):
    x_slice = tf.slice(x, [0, i], x_shape)
    xp = tf.matmul(F, x_slice)
    x = tf.concat([x, xp], 1)
    i = i + 1
    return x, i


shape_invariants = [tf.TensorShape([2, None]), i.get_shape(), k.get_shape()]

loop = tf.while_loop(loop_cond, loop_body, [x, i, k], shape_invariants)


class Model(object):

    def __init__(self, F, C, G, H, x0_dist, w_dist, v_dist):
        pass

    def __define_observations_simulation(self):
        pass

    def __define_likelihood_computation(self):
        pass

    def __isConformable(self):
        pass

    def __isObservable(self):
        pass

    def __isControlable(self):
        pass

    def simulate(self, theta, u, t):
        pass


# TODO: inject theta parameters
def simulate(F, C, G, H, x0, P0, u, t):
    '''arguments are constant tensors'''

    # TODO: check model if model is conformable

    x = tf.contrib.distributions.MultivariateNormalDiag(x0, P0).sample([1])

    def loop_cond(x, P, u, y, i, k):
        pass

    def loop_body(vars):

        def state_propagate(x, t):
            w = w_dist.sample([1])
            x = tf.matmul(F, x) + tf.matmul(C, u) + tf.matmul(G, w)
            return x

        def covariance_propagate(P, t):
            GQ = tf.matmul(G, Q)
            P = tf.matmul(F, P) + tf.matmul(P, F, True) + tf.matmul(GQ, Q, True)
            return P

        # FIXME: t = t[i-1:i]
        xn, info = tf.contrib.integrate.odeint(state_propagate, x, t)

    #


# LL

print('Running graph')

with tf.Session() as sess:
    # writer = tf.summary.FileWriter('.tf', sess.graph)
    xx = sess.run(loop)[0]
    xx = np.transpose(xx)
    print(xx)
    print(sess.run(rv))
    # writer.close()
