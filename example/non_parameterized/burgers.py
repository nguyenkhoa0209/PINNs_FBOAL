import os
import numpy as np
import matplotlib.pyplot as plt
import time
from math import *
import tensorflow as tf
import sys
sys.path.append('../../fboal/')

tf.keras.backend.set_floatx('float32')
tf.get_logger().setLevel('ERROR')

from pinns import PINNs

nx = 256
nt = 100
x = np.linspace(-1, 1, nx)
t = np.linspace(0, 1, nt)
X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
X_star = np.float32(X_star)
X_colloc = X_star

k_f = round(X_colloc.shape[0]/1024)
X_colloc_train = X_colloc[::k_f]

nu_fixed_array = np.linspace(0, 0.025, 100).reshape(-1,1)[10:50][::4]

# reference solution for all nu in nu_fixed_array
u_star_array = np.load('../Data/burgers_sol.npy', allow_pickle=True)

# fix nu
index_nu = 0 # for nu=0.0025
nu_train = nu_fixed_array[index_nu]
u_star = u_star_array[10:50][index_nu]

# define testing points
X_test = X_star[::100]
u_test = u_star_array[10:50][::4][index_nu][::100]

layers = [2] + [50]*4 + [1]
w_pde = 1

lr = 0.0001
thres = 0.02

def net_transform(X_f, model_nn):
    return model_nn(X_f)[:, 0:1] * X_f[:, 1:2] * (X_f[:, 0:1] + 1) * (X_f[:, 0:1] - 1) - tf.math.sin(pi * X_f[:, 0:1])


def f_user(X_f, nu, model_nn):
    x_temp = X_f[:, 0:1]
    t_temp = X_f[:, 1:2]
    nu_temp = nu
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_temp)
        tape.watch(t_temp)

        X_temp = tf.concat([x_temp, t_temp], axis=1)
        u = net_transform(X_temp, model_nn)
        u_x = tape.gradient(u, x_temp)
        u_xx = tape.gradient(u_x, x_temp)
        u_t = tape.gradient(u, t_temp)

    f = u_t + u * u_x - nu_temp * u_xx
    return f

model_classic = PINNs(nu_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                          layers, lr, thres, X_test=X_test, u_test=u_test)
model_classic.train(max_epochs=5000)
#model_classic.save('model_classic', save_format='tf')

model_FBOAML = PINNs(nu_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                         layers, lr, thres, X_test=X_test, u_test=u_test,
                 resampling='FBOAML', period=1000, save_colloc=False, m_FBOAML=10, square_side_FBOAML=0.2)
model_FBOAML.train(max_epochs=5000)
#model_FBOAML.save('model_FBOAML', save_format='tf')


model_RAD = PINNs(nu_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                      layers, lr, thres, X_test=X_test, u_test=u_test,
                 resampling='RAD', period=1000, save_colloc=False, k_RAD=1, c_RAD=1)
model_RAD.train(max_epochs=5000)
#model_RAD.save('model_RAD', save_format='tf')


model_RARD = PINNs(nu_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                       layers, lr, thres, X_test=X_test, u_test=u_test,
                 resampling='RARD', period=1000, save_colloc=False, k_RARD=1, c_RARD=1, m_RARD=5)
model_RARD.train(max_epochs=5000)
#model_RARD.save('model_RARD', save_format='tf')


prediction_classic = net_transform(X_star, model_classic.net_u)
error_classic = np.linalg.norm(prediction_classic - u_star)/np.linalg.norm(u_star)
print('Error by classical PINNs', error_classic)

prediction_FBOAML = net_transform(X_star, model_FBOAML.net_u)
error_FBOAML = np.linalg.norm(prediction_FBOAML - u_star)/np.linalg.norm(u_star)
print('Error by PINNs+FBOAML', error_FBOAML)

prediction_RAD = net_transform(X_star, model_RAD.net_u)
error_RAD = np.linalg.norm(prediction_RAD - u_star)/np.linalg.norm(u_star)
print('Error by PINNs+RAD', error_RAD)

prediction_RARD = net_transform(X_star, model_RARD.net_u)
error_RARD = np.linalg.norm(prediction_RARD - u_star)/np.linalg.norm(u_star)
print('Error by PINNs+RARD', error_RARD)