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
X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
X_star = np.float32(X_star)
X_colloc = X_star

k_f = round(X_colloc.shape[0]/1024)
X_colloc_train = X_colloc[::k_f]

index = 0
k = np.arange(1,3.2,0.2).reshape(-1,1)[index]
k_train = k
u_star = 0.5/np.cosh(2*(X_star[:, 0:1]+np.sqrt(k)*X_star[:, 1:2])) - \
0.5/np.cosh(2*(X_star[:, 0:1]-2*l+np.sqrt(k)*X_star[:, 1:2]))+\
0.5/np.cosh(2*(X_star[:, 0:1]-np.sqrt(k)*X_star[:, 1:2])) - \
0.5/np.cosh(2*(X_star[:, 0:1]+2*l-np.sqrt(k)*X_star[:, 1:2]))

index_init = np.where(X_star[:, 1]==0)
X_star_init = X_star[index_init]
u_star_init = u_star[index_init]
index_left = np.where(X_star[:, 0]==-4)
X_star_left = X_star[index_left]
u_star_left = u_star[index_left]
index_right = np.where(X_star[:, 0]==4)
X_star_right = X_star[index_right]
u_star_right = u_star[index_right]
X_star_bc = np.concatenate((X_star_left, X_star_right), axis=0)
u_star_bc = np.concatenate((u_star_left, u_star_right), axis=0)
u_star_other = np.zeros((X_star_init.shape[0],1))

X_test = X_star[::100]
u_test = u_star[::100]

layers = [2] + [50]*4 + [1]
w_pde = 0.1
lr = 0.0001
thres = 0.005

def net_transform(X_f, model_nn):
    return model_nn(X_f)

def f_user(X_temp, k, model_nn):
    x_temp = X_temp[:, 0:1]
    t_temp = X_temp[:, 1:2]
    k_temp = k
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_temp)
        tape.watch(t_temp)
        X_temp = tf.concat([x_temp, t_temp], axis=1)

        u = net_transform(X_temp, model_nn)[:, 0:1]
        u_x = tape.gradient(u, x_temp)
        u_xx = tape.gradient(u_x, x_temp)
        u_t = tape.gradient(u, t_temp)
        u_tt = tape.gradient(u_t, t_temp)

    f = u_tt - k_temp * u_xx
    return f

def dt_condition(X_neu, model_nn):
    x_temp = X_neu[:, 0:1]
    t_temp = X_neu[:, 1:2]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_temp)
        tape.watch(t_temp)
        X_temp = tf.concat([x_temp, t_temp], axis=1)

        u = net_transform(X_temp, model_nn)[:, 0:1]
        u_t = tape.gradient(u, t_temp)

    return u_t

model_classic = PINNs(k_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                      layers, lr, thres,
                      X_bc=X_star_bc, u_bc=u_star_bc, X_init=X_star_init, u_init=u_star_init,
                      X_other=X_star_init, u_other=u_star_other, net_other=dt_condition, X_test=X_test, u_test=u_test,)
model_classic.train(max_epochs=5000)
#model_classic.save('model_classic', save_format='tf')

model_FBOAML = PINNs(k_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                      layers, lr, thres,
                      X_bc=X_star_bc, u_bc=u_star_bc, X_init=X_star_init, u_init=u_star_init,
                      X_other=X_star_init, u_other=u_star_other, net_other=dt_condition, X_test=X_test, u_test=u_test,
                 resampling='FBOAML', period=1000, save_colloc=False, m_FBOAML=5, square_side_FBOAML=2)
model_FBOAML.train(max_epochs=5000)
#model_FBOAML.save('model_FBOAML', save_format='tf')

model_RAD = PINNs(k_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                      layers, lr, thres,
                      X_bc=X_star_bc, u_bc=u_star_bc, X_init=X_star_init, u_init=u_star_init,
                      X_other=X_star_init, u_other=u_star_other, net_other=dt_condition, X_test=X_test, u_test=u_test,
                 resampling='RAD', period=1000, save_colloc=False, k_RAD=1, c_RAD=1)
model_RAD.train(max_epochs=5000)
#model_RAD.save('model_RAD', save_format='tf')

model_RARD = PINNs(k_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                      layers, lr, thres,
                      X_bc=X_star_bc, u_bc=u_star_bc, X_init=X_star_init, u_init=u_star_init,
                      X_other=X_star_init, u_other=u_star_other, net_other=dt_condition, X_test=X_test, u_test=u_test,
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