print('importing Model from model')
from model import Model
import numpy as np

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
print('creating Model')
m = Model(F, C, G, H, x0_m, x0_c, w_c, v_c, th)

t = np.linspace(0, 5, 25)
u = np.ones([2, 25])
u = u * 10

# run simulation
print('simulating')
rez = m.sim(u, t)
y = rez[1]
L = m.lik(t, u, y)
dL = m.dL(t, u, y)
th0 = [0.9, 0.9]
# th_e = m.mle_fit(th0, t, u, y)
