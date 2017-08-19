
import tensorflow as tf
import control
import numpy as np
from tf.contrib.distributions import MultivariateNormalDiag


class Model(object):

    # TODO: introduce some more default argument values, check types, cast if
    # neccessary
    def __init__(self, F, C, G, H, x0_mean, th, x0_cov, w_cov, v_cov,
                 w_mean=None, v_mean=None):
        """
        Arguments are all callables (functions) of 'th' returning numpy
        matrices except for 'w_mean', 'v_mean' and 'th' itself (of course)
        """
        # TODO: allow both constant matrices and callables

        if w_mean is None:
            w_mean = np.zeros((w_cov(th).shape[0], 1), np.float64)

        if v_mean is None:
            v_mean = np.zeros((v_mean.shape[0], 1), np.float64)

        # check conformability
        u = np.ones((C.shape[0], 1))
        x = np.random.multivariate_normal(x0_mean(th), x0_cov(th))
        w = np.random.multivariate_normal(w_mean, w_cov(th))
        F(th) * x + C(th) * u + G(th) * w
        v = np.random.multivariate_normal(v_mean, v_cov(th))
        H(th) * x + v

        # initialize object
        self.__F = F
        self.__C = C
        self.__G = G
        self.__H = H
        self.__x0_mean = x0_mean
        self.__x0_cov = x0_cov
        self.__w_mean = w_mean
        self.__v_mean = v_mean
        self.__v_cov = v_cov
        self.__th = th

        if not self.__isControllable():
            raise Exception('''Model is not controllable. Set different
                            structure or parameters values''')

        if not self.__isStable():
            raise Exception('''Model is not stable. Set different structure or
                            parameters values''')

        if not self.__isObservable():
            raise Exception('''Model is not observable. Set different  structure
                            or parameters values''')

        # TODO: all instances would share the same graphs, so make graphs class
        # variable
        # self.__define_observations_simulation()
        self.__define_likelihood_computation()

    def __define_observations_simulation(self):

        self.__sim_graph = tf.Graph()
        sim_graph = self.__sim_graph
        s = len(self.__th)
        r = self.__C.shape[1]

        with sim_graph.as_default():
            u = tf.placeholder(tf.float64, shape=[r, 1, None], name='u')
            th = tf.placeholder(tf.float64, shape=[1, s], name='th')
            t = tf.placeholder(tf.float64, shape=[1, None], name='t')
            F = tf.py_func(self.__F, [th], name='F')
            C = tf.py_func(self.__C, [th], name='C')
            G = tf.py_func(self.__G, [th], name='G')
            H = tf.py_func(self.__H, [th], name='H')

            x0_mean = tf.py_func(self.__x0_mean, [th], name='x0_mean')
            x0_cov = tf.py_func(self.__x0_cov, [th], name='x0_cov')
            x0_dist = MultivariateNormalDiag(x0_mean, x0_cov, name='x0_dist')

            Q = tf.py_func(self.__w_cov, [th], name='w_cov')
            w_mean = self.__w_mean
            w_dist = MultivariateNormalDiag(w_mean, Q, name='w_dist')

            R = tf.py_func(self.__v_cov, [th], name='v_cov')
            v_mean = self.__v_mean
            v_dist = MultivariateNormalDiag(v_mean, R, name='v_dist')



            def sim_obs(x):
                v = v_dist.sample([1])
                y = tf.matmul(H, x) + v
                return y

            def sim_loop_cond(x, y, t, k):
                N = t.get_shape()[0]
                return tf.less(k, N-1)

            def sim_loop_body(x, y, t, k):

                u_t_k = tf.slice(u, [0, 0, k], [r, 1, 1])

                def state_propagate(x, t):
                    w = w_dist.sample([1])
                    dx = tf.matmul(F, x) + tf.matmul(C, u_t_k) + tf.matmul(G, w)
                    return dx

                tk = tf.slice(t, k, 2, 'tk')
                x_p = tf.contrib.integrate.odeint(state_propagate, x, tk)

                y_k = sim_obs(x)

                x = tf.stack([x, x_p])
                y = tf.stack([y, y_k])
                k = k + 1

    def __define_likelihood_computation(self):
        def state_propagate(x, t):
            # w = w_dist.sample([1])
            # FIXME: undefined name 'u__t_k'
            # x = tf.matmul(F, x) + tf.matmul(C, u__t_k) + tf.matmul(G, w)
            return x

        def covariance_propagate(P, t):
            # GQtG = tf.matmul(tf.matmul(G, Q), G, transpose_b=True)
            # FIXME: transpose b, not a
            # P = tf.matmul(F, P) + tf.matmul(P, F, transpose_b=True) + GQtG
            return P
        pass



    def __isObservable(self):
        F = self.__F
        C = self.__C
        th = self.__th
        obsv_matrix = control.obsv(F, C)
        rank = np.linalg.matrix_rank(obsv_matrix)
        return rank == F(th).shape[0]

    def __isControllable(self):
        F = self.__F
        C = self.__C
        th = self.__th
        ctrb_matrix = control.ctrb(F(th), C(th))
        rank = np.linalg.matrix_rank(ctrb_matrix)
        return rank == F(th).shape[0]

    def __isStable(self):
        F = self.__F
        th = self.__th
        eigv = np.linalg.eig(F(th))
        real_parts = np.real(eigv)
        return np.all(real_parts < 0)

    def simulate(self, th, u, t):
        # TODO: check controlability, if not cotrolable, print warning
        # TODO: check stability, if not stable, print warning
        # TODO: check observability
        # TODO: run simulation graph
        pass

    def likelihood(self, th, y, u):
        pass

    def mle_fit(self, th, y, u):
        pass
