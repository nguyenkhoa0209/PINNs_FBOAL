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

X_star_xt = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

k_f = round(X_star_xt.shape[0]/1024)
X_colloc_xt = X_star_xt[::k_f]

nu_array = np.linspace(0,0.025,100).reshape(-1,1)
nu_train = nu_array[10:50]
nu_colloc = nu_array[10:50]

X_star = np.concatenate((np.tile(X_star_xt, (nu_colloc.shape[0],1)),
                                       np.repeat(nu_colloc, X_star_xt.shape[0]).reshape(-1,1)), axis=1)
X_colloc_train = np.concatenate((np.tile(X_colloc_xt, (nu_colloc.shape[0],1)),
                                       np.repeat(nu_colloc, X_colloc_xt.shape[0]).reshape(-1,1)), axis=1)

X_test_xt =X_star_xt[::100]
X_test = np.concatenate((np.tile(X_test_xt, (nu_colloc.shape[0],1)),
                                       np.repeat(nu_colloc, X_test_xt.shape[0]).reshape(-1,1)), axis=1)
u_star_array = np.load('../Data/burgers_sol.npy')
u_test = u_star_array[10:50][:, ::100,:]

layers = [3] + [50]*4 + [1]
w_pde = 1
lr = 0.0001
thres = 0.02

@tf.function
def net_transform(X_f, model_nn):
    return model_nn(X_f)[:, 0:1] * X_f[:, 1:2] * (X_f[:, 0:1] + 1) * (X_f[:, 0:1] - 1) - tf.math.sin(pi * X_f[:, 0:1])

@tf.function
def f_user(X_f, nu, model_nn):
    x_temp = X_f[:, 0:1]
    t_temp = X_f[:, 1:2]
    nu_temp = X_f[:, 2:3]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_temp)
        tape.watch(t_temp)
        X_temp = tf.concat([x_temp, t_temp, nu_temp], axis=1)

        u = net_transform(X_temp, model_nn)
        u_x = tape.gradient(u, x_temp)
        u_xx = tape.gradient(u_x, x_temp)
        u_t = tape.gradient(u, t_temp)

    f = u_t + u * u_x - nu_temp * u_xx
    return f

i_test = 0
nu_test = np.linspace(0,0.025,100).reshape(-1,1)[i_test]
nx = 256
nt = 100
x = np.linspace(-1, 1, nx)
t = np.linspace(0, 1, nt)
X, T = np.meshgrid(x, t)
u_star = u_star_array[i_test]
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
X_star_test = np.concatenate((X_star, np.repeat(nu_test, X_star.shape[0]).reshape(-1, 1)), axis=1)

def test_param_classic():
    model_classic = PINNs(nu_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                          layers, lr, thres, X_test=X_test, u_test=u_test)
    model_classic.train(max_epochs=500)
    prediction_classic = net_transform(X_star_test, model_classic.net_u)
    error_classic = np.linalg.norm(prediction_classic - u_star) / np.linalg.norm(u_star)
    assert error_classic < 1

def test_param_fboal():
    model_fboal = PINNs(nu_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                        layers, lr, thres, X_test=X_test, u_test=u_test,
                        resampling='FBOAL', period=100, save_colloc=False, m_FBOAL=5, square_side_FBOAL=0.2)
    model_fboal.train(max_epochs=500)
    prediction_fboal = net_transform(X_star_test, model_fboal.net_u)
    error_fboal = np.linalg.norm(prediction_fboal - u_star) / np.linalg.norm(u_star)
    assert error_fboal < 1

def test_param_rad():
    model_rad = PINNs(nu_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                      layers, lr, thres, X_test=X_test, u_test=u_test,
                      resampling='RAD', period=100, save_colloc=False, k_RAD=1, c_RAD=1)
    model_rad.train(max_epochs=500)
    prediction_rad = net_transform(X_star_test, model_rad.net_u)
    error_rad = np.linalg.norm(prediction_rad - u_star) / np.linalg.norm(u_star)
    assert error_rad < 1

def test_param_rard():
    model_rard = PINNs(nu_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                       layers, lr, thres, X_test=X_test, u_test=u_test,
                       resampling='RARD', period=100, save_colloc=False, k_RARD=1, c_RARD=1, m_RARD=5)
    model_rard.train(max_epochs=500)
    prediction_rard = net_transform(X_star_test, model_rard.net_u)
    error_rard = np.linalg.norm(prediction_rard - u_star) / np.linalg.norm(u_star)
    assert error_rard < 1



