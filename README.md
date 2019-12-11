# Implementation of "Exponential Natural Evolution Strategies" (xNES) 

[https://arxiv.org/abs/1106.4487](https://arxiv.org/abs/1106.4487)

## Usage

    xnes = XNES(f, mu, amat)
    xnes.step(1000)
    print xnes.mu_best 

where

`f` fitness function (real-valued function)

`mu` initial guess of center (scalar or vector)

`amat` initial guess of covariance matrix (scalar or matrix)

See `xnes.py` for a specific example.

*Notes:* 

- Adaptation sampling (`use_adasam=True`) requires tuning the etas a bit to work well. First try without it.
- When using `n_jobs`>1, it is better to turn off multithreading: `export MKL_NUM_THREADS=1`
- `n_jobs`>1 is normally better only if `f` is super expensive.
