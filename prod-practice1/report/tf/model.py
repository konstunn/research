
import tensorflow as tf
import control
import numpy as np
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance


class Model(object):

    # TODO: introduce some more default argument values, check types, cast if
    # neccessary
    def __init__(self, F, C, G, H, x0_mean, x0_cov, w_cov, v_cov, th):
        """
        Arguments are all callables (functions) of 'th' returning numpy
        matrices except for 'th' itself (of course)
        """
        # TODO: evaluate and cast everything to numpy matrices first
        # TODO: cast floats, ints to numpy matrices
        # TODO: allow both constant matrices and callables

        th = np.array(th)

        w_mean = np.zeros((w_cov(th).shape[0], 1), np.float64)
        v_mean = np.zeros((v_cov(th).shape[0], 1), np.float64)

        n = F(th).shape[0]
        m = H(th).shape[0]
        p = G(th).shape[1]

        # check conformability
        u = np.ones((C(th).shape[0], 1))
        # mean must be one dimensional
        x0_m = np.asarray(x0_mean(th)).squeeze()
        x = np.random.multivariate_normal(x0_m, x0_cov(th))
        w = np.random.multivariate_normal(w_mean.squeeze(), w_cov(th))
        v = np.random.multivariate_normal(v_mean.squeeze(), v_cov(th))

        x = x.reshape((n, 1))
        w = w.reshape((p, 1))
        v = v.reshape((m, 1))

        # if model is not conformable, exception would be raised here
        F(th) * x + C(th) * u + G(th) * w
        H(th) * x + v

        # initialize object
        self.__F = F
        self.__C = C
        self.__G = G
        self.__H = H
        self.__x0_mean = x0_mean
        self.__x0_cov = x0_cov
        self.__w_mean = w_mean
        self.__w_cov = w_cov
        self.__v_mean = v_mean
        self.__v_cov = v_cov
        self.__th = th

        self.__validate()

        # TODO: all instances would share the same graphs, so make graphs
        # class-wide variable
        self.__define_observations_simulation()
        # self.__define_likelihood_computation()

    def __define_observations_simulation(self):

        self.__sim_graph = tf.Graph()
        sim_graph = self.__sim_graph
        th = self.__th
        s = len(self.__th)

        # TODO: refactor
        r = self.__C(self.__th).shape[1]
        m = self.__H(self.__th).shape[0]
        n = self.__F(self.__th).shape[1]
        p = self.__G(self.__th).shape[1]

        x0_mean = self.__x0_mean
        x0_cov = self.__x0_cov

        with sim_graph.as_default():

            th = tf.placeholder(tf.float64, shape=[s], name='th')
            N = tf.placeholder(tf.int32, shape=[], name='N')
            u = tf.placeholder(tf.float64, shape=[r, None], name='u')
            t = tf.placeholder(tf.float64, shape=[None], name='t')
            # TODO: check: u.shape[1] == t.shape[0]

            # TODO: refactor
            F = tf.py_func(self.__F, [th], tf.float64, name='F')
            F.set_shape([n, n])
            C = tf.py_func(self.__C, [th], tf.float64, name='C')
            C.set_shape([n, r])
            G = tf.py_func(self.__G, [th], tf.float64, name='G')
            G.set_shape([n, p])
            H = tf.py_func(self.__H, [th], tf.float64, name='H')
            H.set_shape([m, n])

            x0_mean = tf.py_func(x0_mean, [th], tf.float64, name='x0_mean')
            x0_cov = tf.py_func(x0_cov, [th], tf.float64, name='x0_cov')
            x0_dist = MultivariateNormalFullCovariance(x0_mean, x0_cov,
                                                       name='x0_dist')

            Q = tf.py_func(self.__w_cov, [th], tf.float64, name='w_cov')
            w_mean = self.__w_mean
            w_dist = MultivariateNormalFullCovariance(w_mean, Q, name='w_dist')

            R = tf.py_func(self.__v_cov, [th], tf.float64, name='v_cov')
            v_mean = self.__v_mean
            v_dist = MultivariateNormalFullCovariance(v_mean, R, name='v_dist')

            def sim_obs(x):
                v = v_dist.sample([1])
                v = tf.reshape(v, [m, 1])
                y = tf.matmul(H, x) + v
                return y

            def sim_loop_cond(x, y, t, k):
                return tf.less(k, N-1)

            def sim_loop_body(x, y, t, k):

                u_t_k = tf.slice(u, [0, k], [r, 1])

                def state_propagate(x, t):
                    # TODO: reduce code
                    w = w_dist.sample([1])
                    w = tf.reshape(w, [p, 1])
                    # x = tf.reshape(x, [n, 1])
                    Fx = tf.matmul(F, x, name='Fx')
                    Cu = tf.matmul(C, u_t_k, name='Cu')
                    Gw = tf.matmul(G, w, name='Gw')
                    dx = Fx + Cu + Gw
                    # dx = tf.reshape(dx, [n])
                    return dx

                tk = tf.slice(t, [k], [2], 'tk')

                x_k = x[:][-1]
                x_k = tf.reshape(x_k, [n, 1])
                x_k = tf.contrib.integrate.odeint(state_propagate, x_k, tk,
                                                  name='state_propagate')

                x_k = x_k[-1]               # last state (last row)

                y_k = sim_obs(x)

                x = tf.concat([x, x_k], 1)
                y = tf.concat([y, y_k], 1)

                k = k + 1

                return x, y, t, k

            x = x0_dist.sample([1], name='x0_sample')
            x = tf.reshape(x, [n, 1], name='x')

            y = sim_obs(x)
            k = tf.constant(0, name='k')

            shape_invariants = [tf.TensorShape([n, None]),
                                tf.TensorShape([m, None]),
                                t.get_shape(),
                                k.get_shape()]

            sim_loop = tf.while_loop(sim_loop_cond, sim_loop_body,
                                     [x, y, t, k], shape_invariants,
                                     name='sim_loop')

    def __define_likelihood_computation(self):
        def state_propagate(x, t):
            # w = w_dist.sample([1])
            # x = tf.matmul(F, x) + tf.matmul(C, u__t_k) + tf.matmul(G, w)
            return x

        def covariance_propagate(P, t):
            # GQtG = tf.matmul(tf.matmul(G, Q), G, transpose_b=True)
            # P = tf.matmul(F, P) + tf.matmul(P, F, transpose_b=True) + GQtG
            return P
        pass

    def __isObservable(self, th=None):
        if th is None:
            th = self.__th
        F = self.__F
        C = self.__C
        obsv_matrix = control.obsv(F(th), C(th))
        rank = np.linalg.matrix_rank(obsv_matrix)
        return rank == F(th).shape[0]

    def __isControllable(self, th=None):
        if th is None:
            th = self.__th
        F = self.__F
        C = self.__C
        ctrb_matrix = control.ctrb(F(th), C(th))
        rank = np.linalg.matrix_rank(ctrb_matrix)
        return rank == F(th).shape[0]

    def __isStable(self, th=None):
        if th is None:
            th = self.__th
        F = self.__F
        eigv = np.linalg.eigvals(F(th))
        real_parts = np.real(eigv)
        return np.all(real_parts < 0)

    def simulate(self, th, u, t):
        # TODO: check controlability, if not cotrolable, print warning
        # TODO: check stability, if not stable, print warning
        # TODO: check observability
        # TODO: run simulation graph
        pass
    def __validate(self, th=None):
        # TODO: prove, print matrices and their criteria
        if not self.__isControllable(th):
            raise Exception('''Model is not controllable. Set different
                            structure or parameters values''')

        if not self.__isStable(th):
            raise Exception('''Model is not stable. Set different structure or
                            parameters values''')

        if not self.__isObservable(th):
            raise Exception('''Model is not observable. Set different  structure
                            or parameters values''')


    def likelihood(self, th, y, u):
        pass

    def mle_fit(self, th, y, u):
        pass

# test
F = lambda th: np.diag([-.9, -.9])
C = lambda th: np.eye(2)
G = lambda th: np.eye(2)
H = lambda th: np.eye(2)
x0_m = lambda th: np.zeros((2, 1))
x0_c = lambda th: np.diag([.01, .01])
w_c = x0_c
v_c = x0_c
th = [1.0]

m = Model(F, C, G, H, x0_m, x0_c, w_c, v_c, th)
