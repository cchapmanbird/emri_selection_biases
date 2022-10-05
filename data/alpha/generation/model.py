from functools import partial
import numpy as np
from scipy.integrate import quad
from scipy.constants import c
try:
    import cupy as cp
    xp = cp
    from cupyx.scipy.ndimage import map_coordinates
    from cupyx.scipy.special import erf
except ImportError:
    xp = np
    from scipy.ndimage import map_coordinates
    from scipy.special import erf

def trapz(y, x=None, dx=1.0, axis=-1):
    y = xp.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = xp.asanyarray(x)
        if x.ndim == 1:
            d = xp.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = xp.diff(x, axis=axis)
    ndim = y.ndim
    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    product = d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0
    try:
        ret = product.sum(axis)
    except ValueError:
        ret = xp.add.reduce(product, axis)
    return ret

def dL(z):
    '''
    Luminosity distance from redshift assuming default cosmology.
    :param z: Redshift
    :return: dL, in Gpc.
    '''
    h = 0.6774
    omega_m = 0.3089
    omega_lambda = 1 - omega_m
    dH = 1e-5 * c / h

    def E(z):
        return np.sqrt((omega_m * (1 + z) ** (3) + omega_lambda))
    def I(z):
        fact = lambda x: 1 / E(x)
        integral = quad(fact, 0, z)
        return integral[0]

    return (1+z) * dH * I(z) * 1e-3

def truncnorm(xx, mu, sigma, high, low):
    # breakpoint()
    x_op = xp.repeat(xx, mu.size).reshape((xx.size,mu.size)).T
    hi_op = xp.repeat(high, xx.size).reshape((high.size,xx.size))
    lo_op = xp.repeat(low, xx.size).reshape((low.size,xx.size))

    norm = 2**0.5 / np.pi**0.5 / sigma
    norm /= erf((high - mu) / 2**0.5 / sigma) + erf((mu - low) / 2**0.5 / sigma)  #vector of norms
    try:
        prob = xp.exp(-xp.power(xx[None,:] - mu[:,None], 2) / (2 * sigma[:,None]**2)) # array of dims len(xx) * len(mu)
        prob *= norm[:,None]  # should be fine considering dimensionality
        prob[x_op < lo_op] = 0
        prob[x_op > hi_op] = 0
    except IndexError:
        prob = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma**2)) # vector of len(xx)
        prob *= norm
        prob *= (xx <= high) & (xx >= low)
    return prob

def powerlaw(xx, lam, xmin, xmax):
    x_op = xp.repeat(xx, lam.size).reshape((xx.size,lam.size)).T
    hi_op = xp.repeat(xmax, xx.size).reshape((xmax.size,xx.size))
    lo_op = xp.repeat(xmin, xx.size).reshape((xmin.size,xx.size))

    norm = (1+lam)/(xmax**(1+lam) - xmin**(1+lam)) # vector of norms
    try:
        out =  xx[None,:]**lam[:,None] * norm[:,None] # array of dims len(xx) * len(lam)
        out[x_op < lo_op] = 0
        out[x_op > hi_op] = 0
    except IndexError:
        out =  xx**lam * norm # array of dims len(xx) * len(lam)
        out *= (xx <= xmax) & (xx >= xmin)
    return out

def dVdz_over_opz(z):
    h = 0.6774
    omega_m = 0.3089
    omega_lambda = 1 - omega_m
    def E(z):
        return np.sqrt((omega_m * (1 + z) ** (3) + omega_lambda))
    dh = 3000/h # Mpc
    dm = dh * quad(lambda x: 1/E(x),0,z)[0]
    ez = E(z)
    return 4*np.pi * dh * dm**2 / ez / (1+z) # 1+z maps time properly to redshift
