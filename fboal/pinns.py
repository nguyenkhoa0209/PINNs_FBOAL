import os
import numpy as np
import time
from math import *
import tensorflow as tf
from adapt_sampling import FBOAML,RAD,RARD

class PINNs:
    """
        PINNs main class
    """
    def __init__(self, param_pde, X_domain, X_colloc, w_pde, net_transform, net_pde_user, layers, lr, thres,
                 X_bc=None, u_bc=None, X_init=None, u_init=None, X_data=None, u_data=None,
                 X_other=None, u_other=None, net_other=None,X_test=None, u_test=None,
                 resampling=None, period=None, save_colloc=False,
                 m_FBOAML=None, square_side_FBOAML=None, k_RAD=None, c_RAD=None, k_RARD=None, c_RARD=None, m_RARD=None):
        """
        Initialisation function of PINNs non-parametric class

        :param param_pde: parameter of the PDE
        :type param_pde: numpy.ndarray
        :param X_domain: points inside domain to generate new collocation points for adaptive resampling strategy
        :type X_domain: numpy.ndarray
        :param X_colloc: initial collocation points
        :type X_colloc: numpy.ndarray
        :param w_pde: weights for the PDEs residual in the cost function
        :type w_pde: float
        :param net_transform: function to transform the solution output so that it sastisfies automatically some conditions.
        :type net_transform:
        :param net_pde_user: PDE defined by user
        :type net_pde_user:
        :param layers: layers for neural networks
        :type layers: list
        :param lr: learning rate for Adam optimizer
        :type lr: float
        :param thres: threshold to stop the training
        :type thres: float
        :param X_bc: points for boundary conditions
        :type X_bc: numpy.ndarray
        :param u_bc: solution for boundary conditions
        :type u_bc: numpy.ndarray
        :param X_init: points for initial conditions
        :type X_init: numpy.ndarray
        :param u_init: solution for initial conditions
        :type u_init: numpy.ndarray
        :param X_other: points for other boundary conditions
        :type X_other: numpy.ndarray
        :param u_other: other boundary conditions
        :type u_other: numpy.ndarray
        :param net_other: equation for other boundary conditions
        :type net_other:
        :param X_data: points for observed measurements
        :type X_data: numpy.ndarray
        :param u_data: solution for observed measurements
        :type u_data: numpy.ndarray
        :param X_test: testing points
        :type X_test: testing points
        :param u_test: solution for testing points
        :type u_test: numpy.ndarray
        :param resampling: adaptive resampling strategy
        :type resampling: str
        :param period: period of resampling
        :type period: int
        :param save_colloc: option to save collocation points after each period of resampling
        :type save_colloc: bool
        :param m_FBOAML: number of added and removed points in FBOAML
        :type m_FBOAML: int
        :param square_side_FBOAML: side of sub-domains (squares) in FBOAML
        :type square_side_FBOAML: float
        :param k_RAD: hyper-parameter to define collocation points distribution in RAD
        :type k_RAD: float
        :param c_RAD: hyper-parameter to define collocation points distributionin RAD
        :type c_RAD: float
        :param k_RARD: hyper-parameter to define collocation points distributionin RARD
        :type k_RARD: float
        :param c_RARD: hyper-parameter to define collocation points distributionin RARD
        :type c_RARD: float
        :param m_RARD: number of added points in RARD
        :type m_RARD: int

        :returns: Instantiate a PINNs non-parametric caller
        """

        if X_bc is None:
            print("No available data on the boundary")
            self.u_bc = 0
            self.nb_bc = 0
        else:
            self.X_bc = tf.convert_to_tensor(X_bc, dtype='float32')
            self.u_bc = tf.convert_to_tensor(u_bc, dtype='float32')
            self.nb_bc = self.X_bc.shape[0]

        if X_init is None:
            print("No available data at the initial instant")
            self.u_init = 0
            self.nb_init = 0
        else:
            self.X_init = tf.convert_to_tensor(X_init, dtype='float32')
            self.u_init = tf.convert_to_tensor(u_init, dtype='float32')
            self.nb_init = self.X_init.shape[0]

        if X_data is None:
            print('No available data inside the domain')
            self.u_data = 0
            self.nb_data = 0
        else:
            self.X_data = tf.convert_to_tensor(X_data, dtype='float32')
            self.u_data = tf.convert_to_tensor(u_data, dtype='float32')
            self.nb_data = self.X_data.shape[0]

        if X_other is None:
            print("No other condition is provided")
            self.u_other = 0
            self.nb_other = 0
        else:
            self.X_other = tf.convert_to_tensor(X_other, dtype='float32')
            self.u_other = tf.convert_to_tensor(u_other, dtype='float32')
            self.nb_other = self.X_other.shape[0]
            self.net_other = net_other

        if X_test is None:
            print('No available data for testing')
            self.u_test = 0
            self.nb_test = 0
        else:
            self.X_test = tf.convert_to_tensor(X_test, dtype='float32')
            self.u_test = tf.convert_to_tensor(u_test, dtype='float32')
            self.nb_test = self.X_test.shape[0]

        self.X_domain = tf.convert_to_tensor(X_domain, dtype='float32')
        self.X_colloc = tf.convert_to_tensor(X_colloc, dtype='float32')
        self.param_pde = tf.convert_to_tensor(param_pde, dtype='float32')
        self.nb_param = self.param_pde.shape[0]
        self.nb_colloc = self.X_colloc.shape[0]
        self.net_pde_user = net_pde_user
        self.w_pde = w_pde

        if resampling == 'FBOAML':
            if m_FBOAML == None or square_side_FBOAML == None or period == None:
                raise TypeError("Must provide m_FBOAML, square_side_FBOAML and period")
        elif resampling == 'RAD':
            if k_RAD == None or c_RAD == None or period == None:
                raise TypeError("Must provide k_RAD, c_RAD and period")
        elif resampling == 'RARD':
            if k_RARD == None or c_RARD == None or m_RARD == None or period == None:
                raise TypeError("Must provide k_RARD, c_RARD, m_RARD and period")
        elif resampling == None:
            print('The collocation points are fixed during the training')
        else:
            print('The adaptive sampling strategy is not supported')

        self.resampling = resampling
        self.period = period
        self.save_colloc = save_colloc
        self.m_FBOAML = m_FBOAML
        self.square_side_FBOAML = square_side_FBOAML
        self.k_RAD = k_RAD
        self.c_RAD = c_RAD
        self.k_RARD = k_RARD
        self.c_RARD = c_RARD
        self.m_RARD = m_RARD

        self.layers = layers
        self.net_transform = net_transform
        self.net_u = tf.keras.Sequential()
        self.net_u.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
        for i in range(1, len(self.layers) - 1):
            self.net_u.add(
                tf.keras.layers.Dense(self.layers[i], activation=tf.nn.tanh, kernel_initializer="glorot_normal"))
        self.net_u.add(tf.keras.layers.Dense(self.layers[-1], activation=None, kernel_initializer="glorot_normal"))

        self.tf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.loss_array = np.array([])
        self.test_array = np.array([])
        self.thres = thres

    def wrap_training_variables(self):
        """
        Define training parameters in the neural networks

        :return: training parameters
        """
        var = self.net_u.trainable_variables
        return var

    @tf.function
    def net_pde(self, X_f, param_f, model_nn):
        """
        Call PDE function defined by users
        :param X_f: collocation points
        :type X_f: numpy.ndarray
        :param param_f: parameter of the PDE
        :type param_f: float
        :param model_nn: neural networks
        :type model_nn:

        :return: PDEs residual vectors
        """
        f = self.net_pde_user(X_f, param_f, model_nn)
        return f


    @tf.function
    def loss_pinns(self, X_f, param_f, model_nn, u_pred_bc, u_star_bc, u_pred_init, u_star_init, u_pred_data,
                   u_star_data, u_pred_other, u_star_other):
        """
        Define the cost function
        :param X_f: collocation points
        :type X_f: numpy.ndarray
        :param param_f: parameter of the PDE
        :type param_f: float
        :param model_nn: neural networks
        :type model_nn:
        :param u_pred_bc: prediction for the solution on the boundary
        :type u_pred_bc: numpy.ndarray
        :param u_star_bc: reference solution on the boundary
        :type u_star_bc: numpy.ndarray
        :param u_pred_init: prediction for the solution at initial instant
        :type u_pred_init: numpy.ndarray
        :param u_star_init: reference solution at initial instant
        :type u_star_init: numpy.ndarray
        :param u_pred_data: prediction for the observed measurements
        :type u_pred_data: numpy.ndarray
        :param u_star_data: reference solution for the observed measurements
        :type u_star_data: numpy.ndarray
        :param u_pred_other: prediction for the solution on other boundary
        :type u_pred_other: numpy.ndarray
        :param u_star_other: reference solution on other boundary
        :type u_star_other: numpy.ndarray

        :return: loss value during the training
        """
        #f_value = 0
        #if self.nb_colloc > 0:
        #    f = self.net_pde(X_f, param_f, model_nn)
        #    num_pde = len(f)
        #    for i in range(num_pde):
        #        f_value += tf.reduce_mean(tf.square(f[i]))
        loss_obs = 0
        loss_f = 0
        if self.nb_colloc > 0:
            f = self.net_pde(X_f, param_f, model_nn)
        else:
            f = 0
        if self.nb_param == 1:
            loss_f += tf.reduce_mean(tf.square(f))
            loss_obs = 0
            if self.nb_bc > 0:
                loss_obs += tf.reduce_mean(tf.square(u_pred_bc - u_star_bc))
            if self.nb_init > 0:
                loss_obs += tf.reduce_mean(tf.square(u_pred_init - u_star_init))
            if self.nb_data > 0:
                loss_obs += tf.reduce_mean(tf.square(u_pred_data - u_star_data))
            if self.nb_other >0:
                loss_obs += tf.reduce_mean(tf.square(u_pred_other - u_star_other))
        else:
            for i_param in range(self.nb_param):
                if self.nb_bc > 0:
                    size_bc = int(u_star_bc.shape[0] / self.nb_param)
                    loss_obs += tf.reduce_mean(tf.square(
                        u_pred_bc[size_bc * i_param:size_bc * (i_param + 1)] - u_star_bc[size_bc * i_param:size_bc * (
                                    i_param + 1)]))
                if self.nb_init > 0:
                    size_init = int(u_star_init.shape[0] / self.nb_param)
                    loss_obs += tf.reduce_mean(tf.square(
                        u_pred_init[size_init * i_param:size_init * (i_param + 1)] - u_star_init[size_init * i_param:size_init * (
                                    i_param + 1)]))
                if self.nb_data > 0:
                    size_data = int(u_star_data.shape[0] / self.nb_param)
                    loss_obs += tf.reduce_mean(tf.square(
                        u_pred_data[size_data * i_param:size_data * (i_param + 1)] - u_star_data[size_data * i_param:size_data * (
                                    i_param + 1)]))
                if self.nb_other > 0:
                    size_other = int(u_star_other.shape[0] / self.nb_param)
                    loss_obs += tf.reduce_mean(tf.square(
                        u_pred_other[size_other * i_param:size_other * (i_param + 1)] - u_star_other[size_other * i_param:size_other * (
                                    i_param + 1)]))

                index_i_param = tf.where(X_f[:, -1] == param_f[i_param])
                index_i_param = tf.reshape(index_i_param, [-1])
                f_i = tf.gather(f, index_i_param)
                loss_f += tf.reduce_mean(tf.square(f_i))
        loss = loss_obs + loss_f*self.w_pde
        return loss

    @tf.function
    def test_pde(self, X_sup_test, u_sup_test, model_test):
        """
        Define testing function
        :param X_sup_test: testing points
        :type X_sup_test: numpy.ndarray
        :param u_sup_test: reference solution on testing points
        :type u_sup_test: numpy.ndarray
        :param model_test: neural networks
        :type model_test:

        :return: error in testing data set
        """
        u_pred_test = self.net_transform(X_sup_test, model_test)
        return tf.reduce_mean(
            tf.square(u_pred_test - u_sup_test)) / tf.reduce_mean(tf.square(u_sup_test))

    @tf.function
    def get_grad(self, X_f, param_f):
        """
        Calculate the gradients of the cost function w.r.t. training variables
        :param X_f: collocation points
        :type X_f: numpy.ndarray
        :param param_f: parameter of the PDE
        :type param_f: float

        :return: gradients
        """
        with tf.GradientTape() as tape:
            if self.nb_bc > 0:
                u_pred_bc = self.net_transform(self.X_bc, self.net_u)
            else:
                u_pred_bc = 0

            if self.nb_init > 0:
                u_pred_init = self.net_transform(self.X_init, self.net_u)
            else:
                u_pred_init = 0

            if self.nb_data > 0:
                u_pred_data = self.net_transform(self.X_data, self.net_u)
            else:
                u_pred_data = 0

            if self.nb_other > 0:
                u_pred_other = self.net_other(self.X_other, self.net_u)
            else:
                u_pred_other = 0

            loss_value = self.loss_pinns(X_f, param_f, self.net_u, u_pred_bc, self.u_bc, u_pred_init, self.u_init,
                                         u_pred_data, self.u_data, u_pred_other, self.u_other)

        grads = tape.gradient(loss_value, self.wrap_training_variables())

        return loss_value, grads

    def train(self, max_epochs=0):
        """
        Train the neural networks
        :param max_epochs: maximum number of epochs for Adam optimizer
        :type max_epochs: int
        """
        @tf.function
        def train_step(X_f, param_f):
            loss_value, grads = self.get_grad(X_f, param_f)
            self.tf_optimizer.apply_gradients(
                zip(grads, self.net_u.trainable_variables))
            return loss_value

        for epoch in range(max_epochs):
            loss_value = train_step(self.X_colloc, self.param_pde)
            print('Loss pinns at %d epoch:' % epoch, loss_value.numpy())
            self.loss_array = np.append(self.loss_array, loss_value.numpy())
            if self.resampling is not None:
                if ((epoch + 1) % self.period == 0):
                    if self.resampling == 'FBOAML':
                        FBOAML_resampling = FBOAML(self.X_domain.numpy(), self.m_FBOAML, self.square_side_FBOAML, self.X_colloc,
                                                   self.param_pde, self.net_u, self.net_pde)
                        self.X_colloc = FBOAML_resampling.resampling()
                    elif self.resampling == 'RAD':
                        RAD_resampling = RAD(self.X_domain.numpy(), self.k_RAD, self.c_RAD, self.X_colloc, self.param_pde,
                                             self.net_u, self.net_pde_user)
                        self.X_colloc = RAD_resampling.resampling()
                    elif self.resampling == 'RARD':
                        RARD_resampling = RARD(self.X_domain.numpy(), self.k_RARD, self.c_RARD, self.m_RARD, self.X_colloc,
                                               self.param_pde, self.net_u, self.net_pde_user)
                        self.X_colloc = RARD_resampling.resampling()
                    else:
                        print(
                            'The adaptive sampling strategy is not supported, collocation points are fixed during the training')
                    if self.save_colloc == True:
                        np.save('X_colloc_%s_at_%d_epoch' % (self.resampling, epoch + 1), self.X_colloc.numpy())

            if self.X_test is not None:
                if ((epoch + 1) % 1000 == 0):
                    if self.nb_param==1:
                        res_test = self.test_pde(self.X_test, self.u_test, self.net_u)
                        self.test_array = np.append(self.test_array, res_test.numpy())
                        if res_test.numpy() < self.thres**2:
                            break
                    else:
                        res_test_array = np.array([])
                        for i_param in range(self.nb_param):
                            size_test = int(self.u_test.shape[0] / self.nb_param)
                            res_test = self.test_pde(self.X_test[size_test * i_param:size_test * (i_param + 1)],
                                                     self.u_test[size_test * i_param:size_test * (i_param + 1)], self.net_u)
                            res_test_array = np.append(res_test_array, res_test.numpy())
                        if np.mean(res_test_array) < self.thres**2:
                            break