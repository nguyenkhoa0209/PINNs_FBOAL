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

l = 4
T = 5.5
nx = 256
nt = 100
x = np.linspace(-l, l, nx)
t = np.linspace(0, T, nt)
X, T = np.meshgrid(x, t)

X_star_xt = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

k_f = round(X_star_xt.shape[0] / 1024)
X_colloc_xt = X_star_xt[::k_f]

l = 4
k_array = np.arange(0.7, 3.2, 0.05)[6:-3]
k_train = k_array
X_star_array = []
X_colloc_array = []
X_test_array = []
X_star_bc_array = []
X_star_init_array = []
u_test_array = []
u_star_bc_array = []
u_star_init_array = []
for k in k_array:
    # print(k)
    u_star = 0.5 / np.cosh(2 * (X_star_xt[:, 0:1] + np.sqrt(k) * X_star_xt[:, 1:2])) - \
             0.5 / np.cosh(2 * (X_star_xt[:, 0:1] - 2 * l + np.sqrt(k) * X_star_xt[:, 1:2])) + \
             0.5 / np.cosh(2 * (X_star_xt[:, 0:1] - np.sqrt(k) * X_star_xt[:, 1:2])) - \
             0.5 / np.cosh(2 * (X_star_xt[:, 0:1] + 2 * l - np.sqrt(k) * X_star_xt[:, 1:2]))

    X_test_xt = X_star_xt[::100]
    X_test = np.concatenate((np.tile(X_test_xt, (1, 1)),
                             np.repeat(k, X_test_xt.shape[0]).reshape(-1, 1)), axis=1)
    X_test_array.append(X_test)
    u_test = u_star[::100]
    u_test_array.append(u_test)

    index_init = np.where(X_star_xt[:, 1] == 0)
    X_star_init = X_star_xt[index_init]
    u_star_init = u_star[index_init]
    index_left = np.where(X_star_xt[:, 0] == -4)
    X_star_left = X_star_xt[index_left]
    u_star_left = u_star[index_left]
    index_right = np.where(X_star_xt[:, 0] == 4)
    X_star_right = X_star_xt[index_right]
    u_star_right = u_star[index_right]
    X_star_bc = np.concatenate((X_star_left, X_star_right), axis=0)
    u_star_bc = np.concatenate((u_star_left, u_star_right), axis=0)

    X_star_k = np.concatenate((np.tile(X_star_xt, (1, 1)),
                               np.repeat(k, X_star_xt.shape[0]).reshape(-1, 1)), axis=1)

    X_star_bc_k = np.concatenate((np.tile(X_star_bc, (1, 1)),
                                  np.repeat(k, X_star_bc.shape[0]).reshape(-1, 1)), axis=1)
    X_star_init_k = np.concatenate((np.tile(X_star_init, (1, 1)),
                                    np.repeat(k, X_star_init.shape[0]).reshape(-1, 1)), axis=1)

    X_star_array.append(X_star_k)
    X_star_bc_array.append(X_star_bc_k)
    X_star_init_array.append(X_star_init_k)
    u_star_bc_array.append(u_star_bc)
    u_star_init_array.append(u_star_init)

    X_colloc_train = X_star_xt[::int(X_star_xt.shape[0] / 1024)]
    X_colloc_train_k = np.concatenate((np.tile(X_colloc_train, (1, 1)),
                                       np.repeat(k, X_colloc_train.shape[0]).reshape(-1, 1)), axis=1)
    X_colloc_array.append(X_colloc_train_k)

X_star = np.concatenate((X_star_array), axis=0)
X_star_bc = np.concatenate((X_star_bc_array), axis=0)
X_star_init = np.concatenate((X_star_init_array), axis=0)
X_test = np.concatenate((X_test_array), axis=0)
X_colloc_train = np.concatenate((X_colloc_array), axis=0)
X_star_other = X_star_init

u_star_bc = np.concatenate((u_star_bc_array), axis=0)
u_star_init = np.concatenate((u_star_init_array), axis=0)
u_test = np.concatenate((u_test_array), axis=0)
u_star_other = np.zeros((X_star_other.shape[0],))

layers = [3] + [50] * 4 + [1]
w_pde = 1
lr = 0.0001
thres = 0.02

def net_transform(X_f, model_nn):
    return model_nn(X_f)


def f_user(X_temp, k, model_nn):
    x_temp = X_temp[:, 0:1]
    t_temp = X_temp[:, 1:2]
    k_temp = X_temp[:, 2:3]  # nu_fixed_array[args.nu_index]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_temp)
        tape.watch(t_temp)
        X_temp = tf.concat([x_temp, t_temp, t_temp], axis=1)

        u = net_transform(X_temp, model_nn)[:, 0:1]
        u_x = tape.gradient(u, x_temp)
        u_xx = tape.gradient(u_x, x_temp)
        u_t = tape.gradient(u, t_temp)
        u_tt = tape.gradient(u_t, t_temp)

    f = u_tt - k_temp * u_xx
    return f


@tf.function
def dt_condition(X_neu, model_nn):
    x_temp = X_neu[:, 0:1]
    t_temp = X_neu[:, 1:2]
    k_temp = X_neu[:, 2:3]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_temp)
        tape.watch(t_temp)
        X_temp = tf.concat([x_temp, t_temp, k_temp], axis=1)

        u = net_transform(X_temp, model_nn)[:, 0:1]
        u_t = tape.gradient(u, t_temp)

    return u_t

model_classic = PINNs(k_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                      layers, lr, thres,
                      X_bc=X_star_bc, u_bc=u_star_bc, X_init=X_star_init, u_init=u_star_init,
                      X_other=X_star_init, u_other=u_star_other, net_other=dt_condition, X_test=X_test, u_test=u_test,)
model_classic.train(max_epochs=2000)
#model_classic.save('model_classic', save_format='tf')

model_FBOAL = PINNs(k_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                      layers, lr, thres,
                      X_bc=X_star_bc, u_bc=u_star_bc, X_init=X_star_init, u_init=u_star_init,
                      X_other=X_star_init, u_other=u_star_other, net_other=dt_condition, X_test=X_test, u_test=u_test,
                 resampling='FBOAL', period=500, save_colloc=False, m_FBOAL=5, square_side_FBOAL=0.5)
model_FBOAL.train(max_epochs=2000)
#model_FBOAL.save('model_FBOAL', save_format='tf')

model_RAD = PINNs(k_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                      layers, lr, thres,
                      X_bc=X_star_bc, u_bc=u_star_bc, X_init=X_star_init, u_init=u_star_init,
                      X_other=X_star_init, u_other=u_star_other, net_other=dt_condition, X_test=X_test, u_test=u_test,
                 resampling='RAD', period=500, save_colloc=False, k_RAD=1, c_RAD=1)
model_RAD.train(max_epochs=2000)
#model_RAD.save('model_RAD', save_format='tf')


model_RARD = PINNs(k_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                      layers, lr, thres,
                      X_bc=X_star_bc, u_bc=u_star_bc, X_init=X_star_init, u_init=u_star_init,
                      X_other=X_star_init, u_other=u_star_other, net_other=dt_condition, X_test=X_test, u_test=u_test,
                 resampling='RARD', period=500, save_colloc=False, k_RARD=1, c_RARD=1, m_RARD=5)
model_RARD.train(max_epochs=2000)
#model_RARD.save('model_RARD', save_format='tf')

c_test = np.arange(0.7, 3.2, 0.05)

error_classic = np.array([])
error_FBOAL = np.array([])
error_RAD = np.array([])
error_RARD = np.array([])

for i in range(c_test.shape[0]):
    # i=20
    #     print(i)
    #     k = i
    k = c_test[i]
    u_star = 0.5 / np.cosh(2 * (X_star_xt[:, 0:1] + np.sqrt(k) * X_star_xt[:, 1:2])) - \
             0.5 / np.cosh(2 * (X_star_xt[:, 0:1] - 2 * l + np.sqrt(k) * X_star_xt[:, 1:2])) + \
             0.5 / np.cosh(2 * (X_star_xt[:, 0:1] - np.sqrt(k) * X_star_xt[:, 1:2])) - \
             0.5 / np.cosh(2 * (X_star_xt[:, 0:1] + 2 * l - np.sqrt(k) * X_star_xt[:, 1:2]))

    X_star_test = np.concatenate((X_star_xt, np.repeat(k, X_star_xt.shape[0]).reshape(-1, 1)), axis=1)

    out_classic = net_transform(X_star_test, model_classic.net_u)
    out_FBOAL = net_transform(X_star_test, model_FBOAL.net_u)
    out_RAD = net_transform(X_star_test, model_RAD.net_u)
    out_RARD = net_transform(X_star_test, model_RARD.net_u)

    error_classic = np.append(error_classic, np.linalg.norm(out_classic - u_star) / np.linalg.norm(u_star))
    error_FBOAL = np.append(error_FBOAL, np.linalg.norm(out_FBOAL - u_star) / np.linalg.norm(u_star))
    error_RAD = np.append(error_RAD, np.linalg.norm(out_RAD - u_star) / np.linalg.norm(u_star))
    error_RARD = np.append(error_RARD, np.linalg.norm(out_RARD - u_star) / np.linalg.norm(u_star))

print('Error by classical PINNs', error_classic)
print('Error by PINNs+FBOAL', error_FBOAL)
print('Error by PINNs+RAD', error_RAD)
print('Error by PINNs+RARD', error_RARD)