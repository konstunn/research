
import tensorflow as tf
import control
import numpy as np


class Model(object):

    # TODO: introduce some more default argument values, check types, cast if
    # neccessary
    def __init__(self, F, C, G, H, x0_mean, x0_cov, w_mean=None, w_cov,
                 v_mean=None, v_cov, th):
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
        '''arguments are constant tensors'''

        # set method local variables
        th = self.__th
        x0_mean = self.__x0_mean
        x0_cov = self.__x0_cov
        x0_mean = x0_mean(th)
        x0_cov = x0_cov(th)

        x0 = tf.contrib.distributions.MultivariateNormalDiag(x0_mean,
                                                            x0_cov).sample([1])

        def loop_cond(loop_vars):
            pass

        def loop_body(loop_vars):

            def state_propagate(x, t):
                w = w_dist.sample([1])
                x = tf.matmul(F, x) + tf.matmul(C, u) + tf.matmul(G, w)
                return x

            def covariance_propagate(P, t):
                GQ = tf.matmul(G, Q)
                P = tf.matmul(F, P) + tf.matmul(P, F, True) + tf.matmul(GQ, Q,
                                                                        True)
                return P

            # FIXME: t = t[i-1:i]
            xn, info = tf.contrib.integrate.odeint(state_propagate, x, t)

        pass

    def __define_likelihood_computation(self):
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
