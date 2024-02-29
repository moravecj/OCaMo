import numpy as np
from .kernel_correlation import estimate_kernel_correlation

class SGDSchaul:
    def __init__(self, params_trck, theta_init=np.zeros((3,)), delta=np.array([0.001, 0.001, 0.001]), bnd = None) -> None:
        # initialize parameters
        self.theta_ = theta_init.copy()
        self.params_trck_ = params_trck
        self.params_cnt_trck_ = self.theta_.shape[0]
        # intialize arrays for gradient and hessian aggregation over minibatch
        self.g_ = np.zeros((self.params_cnt_trck_,))
        self.g_s_ = np.zeros((self.params_cnt_trck_,))
        self.h_ = np.zeros((self.params_cnt_trck_,))
        # intialize arrays for gradient and hessian aggregation for learning rate estimation
        self.gradients_ = np.zeros((self.params_cnt_trck_,))
        self.gradients_squared_ = np.zeros((self.params_cnt_trck_,))
        self.hessians_ = np.zeros((self.params_cnt_trck_,))
        # initialize the memory size
        self.tau_ = np.ones((self.params_cnt_trck_,))
        
        self.batch_size_ = 0
        self.burn_in_ = 10
        self.delta_ = delta
        self.upd_bnd_ = np.array([0.0024, 0.0024, 0.0024])
        self.mini_batch_ = 1

        self.bnd_ = bnd

    def update(self, pcl, kdtree):
        # estimate function in the paramters and with +/- delta
        f = estimate_kernel_correlation(kdtree, pcl, self.theta_, 9, 10)
        f_pl = np.zeros((self.params_cnt_trck_,))
        for i in self.params_trck_:
            param = self.theta_.copy()
            param[i] += self.delta_[i]
            f_pl[i] = estimate_kernel_correlation(kdtree, pcl, param, 9, 10)
        f_mi = np.zeros((self.params_cnt_trck_,))
        for i in self.params_trck_:
            param = self.theta_.copy()
            param[i] -= self.delta_[i]
            f_mi[i] = estimate_kernel_correlation(kdtree, pcl, param, 9, 10)
        # aggregate minibatch numerical gradients and hessians diagonal
        self.g_ += (f_pl - f_mi) / (2 * self.delta_)
        self.g_s_ += ((f_pl - f_mi) / (2 * self.delta_)) ** 2
        self.h_ += np.abs((f_pl - 2 * f + f_mi) / (self.delta_ ** 2))

        self.batch_size_ += 1
        if self.burn_in_ < 1 and self.mini_batch_ <= self.batch_size_:
            # aggregate overall gradients and hessians
            self.gradients_ = (1 - 1 / self.tau_) * self.gradients_ + (1 / self.tau_) * (self.g_ / self.batch_size_)
            self.gradients_squared_ = (1 - 1 / self.tau_) * self.gradients_squared_ + (1 / self.tau_) * (self.g_s_ / self.batch_size_)
            self.hessians_ = (1 - 1 / self.tau_) * self.hessians_ + (1 / self.tau_) * (self.h_ / self.batch_size_)
            # recalculate memory size
            self.tau_ = (1 - ((self.gradients_ ** 2) / (self.gradients_squared_ + 0.0000001))) * self.tau_ + 1
            self.tau_[np.abs(self.tau_) > 5] = 5
            # estimate the update
            nu = self.g_ / self.hessians_
            # bound the update
            nu[np.abs(nu) > self.upd_bnd_] = self.upd_bnd_[np.abs(nu) > self.upd_bnd_] * (nu[np.abs(nu) > self.upd_bnd_] / np.abs(nu[np.abs(nu) > self.upd_bnd_]))
            # estimate the learning rate
            lr = ((self.gradients_ ** 2) / (self.gradients_squared_ + 0.0000001))
            # update parameters
            self.theta_[self.params_trck_] -= lr[self.params_trck_] * nu[self.params_trck_]
            # clip the parameters to the bound, if necessary
            if not(self.bnd_ is None):
                th = self.theta_[:3]
                th[np.abs(th) > self.bnd_] = self.bnd_[np.abs(th) > self.bnd_] * (th[np.abs(th) > self.bnd_] / np.abs(th[np.abs(th) > self.bnd_]))
                self.theta_[:3] = th 
                
            self.batch_size_ = 0
            self.g_[:] = 0
            self.g_s_[:] = 0
            self.h_[:] = 0

        self.burn_in_ -= 1
