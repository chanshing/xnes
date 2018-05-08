"""
xNES from 'Natural Evolution Strategies'
if n_jobs>1, I suggest using "export MKL_NUM_THREADS=1"
"""
import joblib
import random
import numpy as np

import scipy as sp
from scipy import (dot, eye, randn, asarray, array, trace, log, exp, sqrt, mean, sum, argsort, square, arange)
from scipy.stats import multivariate_normal, norm
from scipy.linalg import (det, expm)

class XNES(object):
    def __init__(self, f, mu, amat,
                 eta_mu=1.0, eta_sigma=None, eta_bmat=None,
                 npop=None, use_fshape=True, use_adasam=False, patience=100, n_jobs=1):
        self.f = f
        self.mu = mu
        self.eta_mu = eta_mu
        self.use_adasam = use_adasam
        self.n_jobs = n_jobs

        dim = len(mu)
        sigma = abs(det(amat))**(1.0/dim)
        bmat = amat*(1.0/sigma)
        self.dim = dim
        self.sigma = sigma
        self.bmat = bmat

        # default population size and learning rates
        npop = int(4 + 3*log(dim)) if npop is None else npop
        eta_sigma = 3*(3+log(dim))*(1.0/(5*dim*sqrt(dim))) if eta_sigma is None else eta_sigma
        eta_bmat = 3*(3+log(dim))*(1.0/(5*dim*sqrt(dim))) if eta_bmat is None else eta_bmat
        self.npop = npop
        self.eta_sigma = eta_sigma
        self.eta_bmat = eta_bmat

        # compute utilities if using fitness shaping
        if use_fshape:
            a = log(1+0.5*npop)
            utilities = array([max(0, a-log(k)) for k in range(1,npop+1)])
            utilities /= sum(utilities)
            utilities -= 1.0/npop           # broadcast
            utilities = utilities[::-1]  # ascending order
        else:
            utilities = None
        self.use_fshape = use_fshape
        self.utilities = utilities

        # stuff for adasam
        self.eta_sigma_init = eta_sigma
        self.sigma_old = None

        # logging
        self.fitness_best = None
        self.mu_best = None
        self.done = False
        self.counter = 0
        self.patience = patience
        self.history = {'eta_sigma':[], 'sigma':[], 'fitness':[]}

        # do not use these when hill-climbing
        if npop == 1:
            self.use_fshape = False
            self.use_adasam = False

    def step(self, niter):
        """ xNES """
        f = self.f
        mu, sigma, bmat = self.mu, self.sigma, self.bmat
        eta_mu, eta_sigma, eta_bmat = self.eta_mu, self.eta_sigma, self.eta_bmat
        npop = self.npop
        dim = self.dim
        sigma_old = self.sigma_old

        eyemat = eye(dim)

        with joblib.Parallel(n_jobs=self.n_jobs) as parallel:

            for i in range(niter):
                s_try = randn(npop, dim)
                z_try = mu + sigma * dot(s_try, bmat)     # broadcast

                f_try = parallel(joblib.delayed(f)(z) for z in z_try)
                f_try = asarray(f_try)

                # save if best
                fitness = mean(f_try)
                if fitness - 1e-8 > self.fitness_best:
                    self.fitness_best = fitness
                    self.mu_best = mu.copy()
                    self.counter = 0
                else: self.counter += 1
                if self.counter > self.patience:
                    self.done = True
                    return

                isort = argsort(f_try)
                f_try = f_try[isort]
                s_try = s_try[isort]
                z_try = z_try[isort]

                u_try = self.utilities if self.use_fshape else f_try

                if self.use_adasam and sigma_old is not None:  # sigma_old must be available
                    eta_sigma = self.adasam(eta_sigma, mu, sigma, bmat, sigma_old, z_try)

                dj_delta = dot(u_try, s_try)
                dj_mmat = dot(s_try.T, s_try*u_try.reshape(npop,1)) - sum(u_try)*eyemat
                dj_sigma = trace(dj_mmat)*(1.0/dim)
                dj_bmat = dj_mmat - dj_sigma*eyemat

                sigma_old = sigma

                # update
                mu += eta_mu * sigma * dot(bmat, dj_delta)
                sigma *= exp(0.5 * eta_sigma * dj_sigma)
                bmat = dot(bmat, expm(0.5 * eta_bmat * dj_bmat))

                # logging
                self.history['fitness'].append(fitness)
                self.history['sigma'].append(sigma)
                self.history['eta_sigma'].append(eta_sigma)

        # keep last results
        self.mu, self.sigma, self.bmat = mu, sigma, bmat
        self.eta_sigma = eta_sigma
        self.sigma_old = sigma_old

    def adasam(self, eta_sigma, mu, sigma, bmat, sigma_old, z_try):
        """ Adaptation sampling """
        eta_sigma_init = self.eta_sigma_init
        dim = self.dim
        c = .1
        rho = 0.5 - 1./(3*(dim+1))  # empirical

        bbmat = dot(bmat.T, bmat)
        cov = sigma**2 * bbmat
        sigma_ = sigma * sqrt(sigma*(1./sigma_old))  # increase by 1.5
        cov_ = sigma_**2 * bbmat

        p0 = multivariate_normal.logpdf(z_try, mean=mu, cov=cov)
        p1 = multivariate_normal.logpdf(z_try, mean=mu, cov=cov_)
        w = exp(p1-p0)

        # Mann-Whitney. It is assumed z_try was in ascending order.
        n = self.npop
        n_ = sum(w)
        u_ = sum(w * (arange(n)+0.5))

        u_mu = n*n_*0.5
        u_sigma = sqrt(n*n_*(n+n_+1)/12.)
        cum = norm.cdf(u_, loc=u_mu, scale=u_sigma)

        if cum < rho:
            return (1-c)*eta_sigma + c*eta_sigma_init
        else:
            return min(1, (1+c)*eta_sigma)


if __name__ == '__main__':
    import time
    np.random.seed(42)
    random.seed(42)

    def f(x):                   # sin(x^2+y^2)/(x^2+y^2)
        r = sum(square(x))
        return sp.sin(r)/r

    mu = array([9999.,-9999.])  # a bad init guess
    amat = eye(2)

    # when adasam, use conservative eta
    xnes = XNES(f, mu, amat, npop=50, use_adasam=True, eta_bmat=0.01, eta_sigma=.1, patience=9999)
    t0 = time.time()

    for i in range(20):
        xnes.step(100)
        print "Current: ({},{})".format(*xnes.mu)

    print("Exact solution is (0,0)")
    print("Took {} secs".format(time.time()-t0))

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3,1)
    axs[0].plot(xnes.history['fitness'])
    axs[1].plot(xnes.history['sigma'])
    axs[2].plot(xnes.history['eta_sigma'])
    axs[0].set_ylabel('fitness')
    axs[1].set_ylabel(r'$\sigma$')
    axs[2].set_ylabel(r'$\eta_{\sigma}$')
    fig.show()
