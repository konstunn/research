
import math
import tensorflow as tf
import control
import numpy as np
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
import scipy


class Model(object):

    # TODO: introduce some more default argument values, check types, cast if
    # neccessary
    def __init__(self, F, C, G, H, x0_mean, x0_cov, w_cov, v_cov, th):
        """
        Arguments are all callables (functions) of 'th' returning python lists
        except for 'th' itself (of course)
        """

        # TODO: evaluate and cast everything to numpy matrices first
        # TODO: cast floats, ints to numpy matrices
        # TODO: allow both constant matrices and callables

        # store arguments, after that check them
        self.__F = F
        self.__C = C
        self.__G = G
        self.__H = H
        self.__x0_mean = x0_mean
        self.__x0_cov = x0_cov
        self.__w_cov = w_cov
        self.__v_cov = v_cov
        self.__th = th

        # evaluate all functions
        th = np.array(th)
        F = np.array(F(th))
        C = np.array(C(th))
        H = np.array(H(th))
        G = np.array(G(th))
        w_cov = np.array(w_cov(th))    # Q
        v_cov = np.array(v_cov(th))    # R
        x0_m = np.array(x0_mean(th))
        x0_cov = np.array(x0_cov(th))  # P_0

        # get dimensions and store them as well
        self.__n = n = F.shape[0]
        self.__m = m = H.shape[0]
        self.__p = p = G.shape[1]
        self.__r = r = C.shape[1]

        # generate means
        w_mean = np.zeros([p, 1], np.float64)
        v_mean = np.zeros([m, 1], np.float64)

        # and store them
        self.__w_mean = w_mean
        self.__v_mean = v_mean

        # check conformability
        u = np.ones([r, 1])
        # generate random vectors
        # squeeze, because mean must be one dimensional
        x = np.random.multivariate_normal(x0_m.squeeze(), x0_cov)
        w = np.random.multivariate_normal(w_mean.squeeze(), w_cov)
        v = np.random.multivariate_normal(v_mean.squeeze(), v_cov)

        # shape them as column-vectors
        x = x.reshape([n, 1])
        w = w.reshape([p, 1])
        v = v.reshape([m, 1])

        # if model is not conformable, exception would be raised (thrown) here
        F * x + C * u + G * w
        H * x + v

        # check controllability, stability, observability
        self.__validate()

        # if the execution reached here, all is fine so
        # define corresponding computational tensorflow graphs
        self.__define_observations_simulation()
        self.__define_likelihood_computation()

    def __define_observations_simulation(self):
        # TODO: reduce code not to create extra operations

        self.__sim_graph = tf.Graph()
        sim_graph = self.__sim_graph

        r = self.__r
        m = self.__m
        n = self.__n
        p = self.__p

        x0_mean = self.__x0_mean
        x0_cov = self.__x0_cov

        with sim_graph.as_default():

            th = tf.placeholder(tf.float64, shape=[None], name='th')

            # TODO: this should be continuous function of time
            # but try to let pass array also
            u = tf.placeholder(tf.float64, shape=[r, None], name='u')

            t = tf.placeholder(tf.float64, shape=[None], name='t')

            # TODO: refactor

            # FIXME: gradient of py_func is None
            # TODO: embed function itself in the graph, must rebuild the graph
            # if the structure of the model change
            # use tf.convert_to_tensor
            F = tf.convert_to_tensor(self.__F(th), tf.float64)
            F.set_shape([n, n])

            C = tf.convert_to_tensor(self.__C(th), tf.float64)
            C.set_shape([n, r])

            G = tf.convert_to_tensor(self.__G(th), tf.float64)
            G.set_shape([n, p])

            H = tf.convert_to_tensor(self.__H(th), tf.float64)
            H.set_shape([m, n])

            x0_mean = tf.convert_to_tensor(x0_mean(th), tf.float64)
            x0_mean = tf.squeeze(x0_mean)

            x0_cov = tf.convert_to_tensor(x0_cov(th), tf.float64)
            x0_cov.set_shape([n, n])

            x0_dist = MultivariateNormalFullCovariance(x0_mean, x0_cov,
                                                       name='x0_dist')

            Q = tf.convert_to_tensor(self.__w_cov(th), tf.float64)
            Q.set_shape([p, p])

            w_mean = self.__w_mean.squeeze()
            w_dist = MultivariateNormalFullCovariance(w_mean, Q, name='w_dist')

            R = tf.convert_to_tensor(self.__v_cov(th), tf.float64)
            R.set_shape([m, m])
            v_mean = self.__v_mean.squeeze()
            v_dist = MultivariateNormalFullCovariance(v_mean, R, name='v_dist')

            def sim_obs(x):
                v = v_dist.sample()
                v = tf.reshape(v, [m, 1])
                y = H @ x + v  # the syntax is valid for Python >= 3.5
                return y

            def sim_loop_cond(x, y, t, k):
                N = tf.stack([tf.shape(t)[0]])
                N = tf.reshape(N, ())
                return tf.less(k, N-1)

            def sim_loop_body(x, y, t, k):

                # TODO: this should be function of time
                u_t_k = tf.slice(u, [0, k], [r, 1])

                def state_propagate(x, t):
                    w = w_dist.sample()
                    w = tf.reshape(w, [p, 1])
                    Fx = tf.matmul(F, x, name='Fx')
                    Cu = tf.matmul(C, u_t_k, name='Cu')
                    Gw = tf.matmul(G, w, name='Gw')
                    dx = Fx + Cu + Gw
                    return dx

                tk = tf.slice(t, [k], [2], 'tk')

                x_k = x[:, -1]
                x_k = tf.reshape(x_k, [n, 1])

                # decreased default tolerance to avoid max_num_steps exceeded
                # error may increase max_num_steps as an alternative
                x_k = tf.contrib.integrate.odeint(state_propagate, x_k, tk,
                                                  name='state_propagate',
                                                  rtol=1e-4,
                                                  atol=1e-10)

                x_k = x_k[-1]   # last state (last row)

                y_k = sim_obs(x_k)

                # TODO: stack instead of concat
                x = tf.concat([x, x_k], 1)
                y = tf.concat([y, y_k], 1)

                k = k + 1

                return x, y, t, k

            x = x0_dist.sample(name='x0_sample')
            x = tf.reshape(x, [n, 1], name='x')

            # this zeroth measurement should be thrown away
            y = sim_obs(x)
            k = tf.constant(0, name='k')

            shape_invariants = [tf.TensorShape([n, None]),
                                tf.TensorShape([m, None]),
                                t.get_shape(),
                                k.get_shape()]

            sim_loop = tf.while_loop(sim_loop_cond, sim_loop_body,
                                     [x, y, t, k], shape_invariants,
                                     name='sim_loop')

            self.__sim_loop_op = sim_loop

    # defines graph
    def __define_likelihood_computation(self):

        self.__lik_graph = tf.Graph()
        lik_graph = self.__lik_graph

        r = self.__r
        m = self.__m
        n = self.__n
        p = self.__p

        x0_mean = self.__x0_mean
        x0_cov = self.__x0_cov

        with lik_graph.as_default():
            # FIXME: Don't Repeat Yourself (in simulation and here)
            th = tf.placeholder(tf.float64, shape=[None], name='th')
            u = tf.placeholder(tf.float64, shape=[r, None], name='u')
            t = tf.placeholder(tf.float64, shape=[None], name='t')
            y = tf.placeholder(tf.float64, shape=[m, None], name='y')

            N = tf.stack([tf.shape(t)[0]])
            N = tf.reshape(N, ())

            F = tf.convert_to_tensor(self.__F(th), tf.float64)
            F.set_shape([n, n])

            C = tf.convert_to_tensor(self.__C(th), tf.float64)
            C.set_shape([n, r])

            G = tf.convert_to_tensor(self.__G(th), tf.float64)
            G.set_shape([n, p])

            H = tf.convert_to_tensor(self.__H(th), tf.float64)
            H.set_shape([m, n])

            x0_mean = tf.convert_to_tensor(x0_mean(th), tf.float64)
            x0_mean.set_shape([n, 1])

            P_0 = tf.convert_to_tensor(x0_cov(th), tf.float64)
            P_0.set_shape([n, n])

            Q = tf.convert_to_tensor(self.__w_cov(th), tf.float64)
            Q.set_shape([p, p])

            R = tf.convert_to_tensor(self.__v_cov(th), tf.float64)
            R.set_shape([m, m])

            I = tf.eye(n, n, dtype=tf.float64)

            def lik_loop_cond(k, P, S, t, u, x, y):
                return tf.less(k, N-1)

            def lik_loop_body(k, P, S, t, u, x, y):

                # TODO: this should be function of time
                u_t_k = tf.slice(u, [0, k], [r, 1])

                # k+1, cause zeroth measurement should not be taken into account
                y_k = tf.slice(y, [0, k+1], [m, 1])

                t_k = tf.slice(t, [k], [2], 't_k')

                # TODO: extract Kalman filter to a separate class
                def state_predict(x, t):
                    Fx = tf.matmul(F, x, name='Fx')
                    Cu = tf.matmul(C, u_t_k, name='Cu')
                    dx = Fx + Cu
                    return dx

                def covariance_predict(P, t):
                    GQtG = tf.matmul(G @ Q, G, transpose_b=True)
                    PtF = tf.matmul(P, F, transpose_b=True)
                    dP = tf.matmul(F, P) + PtF + GQtG
                    return dP

                x = tf.contrib.integrate.odeint(state_predict, x, t_k,
                                                name='state_predict')
                x = x[-1]

                P = tf.contrib.integrate.odeint(covariance_predict, P, t_k,
                                                name='covariance_predict')
                P = P[-1]

                E = y_k - tf.matmul(H, x)

                B = tf.matmul(H @ P, H, transpose_b=True) + R
                invB = tf.matrix_inverse(B)

                K = tf.matmul(P, H, transpose_b=True) @ invB

                S_k = tf.matmul(E, invB @ E, transpose_a=True)
                S_k = 0.5 * (S_k + tf.log(tf.matrix_determinant(B)))

                S = S + S_k

                # state update
                x = x + tf.matmul(K, E)

                # covariance update
                P = (I - K @ H) @ P

                k = k + 1

                return k, P, S, t, u, x, y

            k = tf.constant(0, name='k')
            P = P_0
            S = tf.constant(0.0, dtype=tf.float64, shape=[1, 1], name='S')
            x = x0_mean

            # TODO: make a named tuple of named list
            lik_loop = tf.while_loop(lik_loop_cond, lik_loop_body,
                                     [k, P, S, t, u, x, y], name='lik_loop')

            dS = tf.gradients(lik_loop[2], th)

            self.__lik_loop_op = lik_loop
            self.__dS = dS

    def __isObservable(self, th=None):
        if th is None:
            th = self.__th
        F = np.array(self.__F(th))
        C = np.array(self.__C(th))
        n = self.__n
        obsv_matrix = control.obsv(F, C)
        rank = np.linalg.matrix_rank(obsv_matrix)
        return rank == n

    def __isControllable(self, th=None):
        if th is None:
            th = self.__th
        F = np.array(self.__F(th))
        C = np.array(self.__C(th))
        n = self.__n
        ctrb_matrix = control.ctrb(F, C)
        rank = np.linalg.matrix_rank(ctrb_matrix)
        return rank == n

    def __isStable(self, th=None):
        if th is None:
            th = self.__th
        F = np.array(self.__F(th))
        eigv = np.linalg.eigvals(F)
        real_parts = np.real(eigv)
        return np.all(real_parts < 0)

    def __validate(self, th=None):
        # FIXME: do not raise exceptions
        # TODO: prove, print matrices and their criteria
        if not self.__isControllable(th):
            # raise Exception('''Model is not controllable. Set different
            #                structure or parameters values''')
            pass

        if not self.__isStable(th):
            # raise Exception('''Model is not stable. Set different structure or
            #                parameters values''')
            pass

        if not self.__isObservable(th):
            # raise Exception('''Model is not observable. Set different
            #                structure or parameters values''')
            pass

    def sim(self, u, t, th=None):
        if th is None:
            th = self.__th

        self.__validate(th)
        g = self.__sim_graph

        if t.shape[0] != u.shape[1]:
            raise Exception('''t.shape[0] != u.shape[1]''')

        # run simulation graph
        with tf.Session(graph=g) as sess:
            t_ph = g.get_tensor_by_name('t:0')
            th_ph = g.get_tensor_by_name('th:0')
            u_ph = g.get_tensor_by_name('u:0')
            rez = sess.run(self.__sim_loop_op, {th_ph: th, t_ph: t, u_ph: u})

        return rez

    def lik(self, t, u, y, th=None):
        if th is None:
            th = self.__th

        # to numpy 1D array
        th = np.array(th).squeeze()

        #self.__validate(th)
        g = self.__lik_graph

        if t.shape[0] != u.shape[1]:
            raise Exception('''t.shape[0] != u.shape[1]''')

        # run lik graph
        with tf.Session(graph=g) as sess:
            t_ph = g.get_tensor_by_name('t:0')
            th_ph = g.get_tensor_by_name('th:0')
            u_ph = g.get_tensor_by_name('u:0')
            y_ph = g.get_tensor_by_name('y:0')
            rez = sess.run(self.__lik_loop_op, {th_ph: th, t_ph: t, u_ph: u,
                                                y_ph: y})

        N = len(t)
        m = y.shape[0]
        S = rez[2]
        S = S + N*m * 0.5 + np.log(2*math.pi)

        return S

    def __L(self, th, t, u, y):
        return self.lik(t, u, y, th)

    def __dL(self, th, t, u, y):
        return self.dL(t, u, y, th)

    def dL(self, t, u, y, th=None):
        if th is None:
            th = self.__th

        # to 1D numpy array
        th = np.array(th).squeeze()

        #self.__validate(th)
        g = self.__lik_graph

        if t.shape[0] != u.shape[1]:
            raise Exception('''t.shape[0] != u.shape[1]''')

        # run lik graph
        with tf.Session(graph=g) as sess:
            t_ph = g.get_tensor_by_name('t:0')
            th_ph = g.get_tensor_by_name('th:0')
            u_ph = g.get_tensor_by_name('u:0')
            y_ph = g.get_tensor_by_name('y:0')
            rez = sess.run(self.__dS, {th_ph: th, t_ph: t, u_ph: u, y_ph: y})

        return rez[0]

    def mle_fit(self, th, t, u, y):
        # TODO: call slsqp
        th0 = th
        th = scipy.optimize.minimize(self.__L, th0, args=(t, u, y),
                                     jac=self.__dL)
        return th

# test

# model
F = lambda th: [[-th[0], 0.],
                [0., -th[1]]]

C = lambda th: [[1.0, 0.],
                [0., 1.0]]

G = lambda th: [[1.0, 0.],
                [0., 1.0]]

H = lambda th: [[1.0, 0.],
                [0., 1.0]]

x0_m = lambda th: [[0.],
                   [0.]]

x0_c = lambda th: [[1e-3, 0.],
                   [0., 1e-3]]
w_c = x0_c
v_c = x0_c

th = [1.0, 1.0]

# TODO: check if there are extra components in 'th'
# TODO: measure model creation time
m = Model(F, C, G, H, x0_m, x0_c, w_c, v_c, th)

t = np.linspace(0, 10, 100)
u = np.ones([2, 100])
u = u * 10

# run simulation
rez = m.sim(u, t)
y = rez[1]
L = m.lik(t, u, y)
dL = m.dL(t, u, y)
th0 = [0.4, 0.5]
print('identificating')
th_e = m.mle_fit(th0, t, u, y)
