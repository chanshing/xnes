# Implementation of "Exponential Natural Evolution Strategies" (xNES) [http://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf](http://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf)

    xnes = XNES(f, mu, amat)
    xnes.step(1000)
    print xnes.mu_best 

where

`f` fitness function (real-valued function)

`mu` initial guess of center (scalar or vector)

`amat` initial guess of covariance matrix (scalar or matrix)

See `xnes.py` for a specific example.

*Notes:* Adaptative sampling (`use_adasam=True`) requires tuning the etas a bit to work well. First try without it.
