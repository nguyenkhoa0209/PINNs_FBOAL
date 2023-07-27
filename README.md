# FBOAL
Fixed-Burget Online Adaptive Learning (FBOAL) for Physics-Informed Neural Networks (PINNs)

The data and code for the paper [Nguyen, T.N.K., Dairay, T., Meunier, R., Millet, C. and Mougeot, M., 2023, June. Fixed-Budget Online Adaptive Learning for Physics-Informed Neural Networks. Towards Parameterized Problem Inference. In International Conference on Computational Science (pp. 453-468). Cham: Springer Nature Switzerland](https://link.springer.com/chapter/10.1007/978-3-031-36027-5_36)

## Basic usage

We provide a package for calling PINNs model with (or without) adaptive sampling methods (FBOAL, RARD, RAD).

To install the package:
```
!git clone https://github.com/nguyenkhoa0209/PINNs_FBOAL
from fboal import pinns, adapt_sampling
from pinns import *
from adapt_sampling import *
```
To call a PINNs model:
```
model = PINNs(param_pde, X_domain, X_colloc, w_pde, net_transform, net_pde_user, layers, lr, thres,
                 X_bc=None, u_bc=None, X_init=None, u_init=None, X_data=None, u_data=None,
                 X_other=None, u_other=None, net_other=None,X_test=None, u_test=None,
                 resampling=None, period=None, save_colloc=False,
                 m_FBOAML=None, square_side_FBOAML=None, k_RAD=None, c_RAD=None, k_RARD=None, c_RARD=None, m_RARD=None)
```

#### Example on Burgers equation

We take an example on Burgers equation and consider the non-parametric case (i.e. the parameter of the PDE is fixed):

<img align="center" src="https://user-images.githubusercontent.com/50335341/225834360-ec2c9894-794d-4644-9a17-4e0bd6cf1e59.png" width="400" height="300">


First, we define the PDE:

```
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
```
where `net_transform` is a function to transform the output of the neural networks so that it sastisfies some conditions. For example, to force the BC/IC to be automatically sastified:
```
def net_transform(X_f, model_nn):
    return model_nn(X_f)[:, 0:1] * X_f[:, 1:2] * (X_f[:, 0:1] + 1) * (X_f[:, 0:1] - 1) - tf.math.sin(pi * X_f[:, 0:1])
```
Then we define the supervised points (optional) and initialize the collocation points and call PINNs:
```
model = PINNs(nu_train, X_star, X_colloc_train, w_pde, net_transform, f_user,
                         layers, lr, thres, X_test=X_test, u_test=u_test,
                 resampling='FBOAML', period=1000, save_colloc=False, m_FBOAML=10, square_side_FBOAML=0.2)
model.train(max_epochs=5000)
```
The colloctions points during the training will be adaptively located where there is important error for the PDE residuals:

<img align="center" src="https://user-images.githubusercontent.com/50335341/226867743-7423eb40-92b7-4b6a-8e86-c9dc504f8c1c.gif" width="400" height="350">


For the parametric case where the PDE parameter is considered as an input of PINNs, the syntax remains the same with little modification in the `f_user` as now `nu_temp=X_f[:, 2:3]`. The total number of collocation points is fixed during the training, however, for each value of the PDE parameter, the distribution and the number of these points can be varied.

<img align="center" src="https://user-images.githubusercontent.com/50335341/226868112-940e1ec5-2629-428e-bb94-2b06b46eb913.gif" width="400" height="300">


## Cite this work

If you use the method or code for academic research, you are encouraged to cite the following paper:

```
@inproceedings{nguyen2023fixed,
  title={Fixed-Budget Online Adaptive Learning for Physics-Informed Neural Networks. Towards Parameterized Problem Inference},
  author={Nguyen, Thi Nguyen Khoa and Dairay, Thibault and Meunier, Rapha{\"e}l and Millet, Christophe and Mougeot, Mathilde},
  booktitle={International Conference on Computational Science},
  pages={453--468},
  year={2023},
  organization={Springer}
}
```