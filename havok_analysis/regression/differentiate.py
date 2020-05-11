'''

      Project: HAVOK analysis
         File: differentiate.py
 File Created: 04.03.2020
       Author: Jan Zetzmann (yangntze+github_havok@gmail.com)
-----
Last Modified: 11.05.2020 21:47:43
  Modified By: Jan Zetzmann (yangntze+github_havok@gmail.com)
-----
    Copyright: 2020 havokanalysisteam

'''


import numpy as np
from scipy import ndimage
from scipy.interpolate import UnivariateSpline

def diff(t_vec, x_vec):
    # if type(vec) == np.array :
    #     pass
    # else:
    if t_vec.ndim!=1:
        raise TypeError('Multi-dimensional time huh? fancy stuff!')

    if x_vec.ndim!=1:
        print('Warning: the vector given to diff is not one dimensional')
        if x_vec[0].ndim==1:
            print('using just the first inner element')
            x_vec = x_vec[0]
        else:
            raise TypeError('Not a one-dimensional (neither first nor second axis) vector!')

    if(x_vec.ndim != t_vec.ndim):
        raise TypeError('Array lengths do not match!')

    dx = np.zeros(np.shape(x_vec))
    # forward difference for first point
    dx[0] = (x_vec[1] - x_vec[0])/(t_vec[1]-t_vec[0])
    # backward difference for last point
    dx[-1] = (x_vec[-1] - x_vec[-2])/(t_vec[-1]-t_vec[-2])
    # central difference for all inner points
    for k in range(1, len(x_vec)-1):
        dx[k] = (x_vec[k+1] - x_vec[k-1])/(t_vec[k+1]-t_vec[k-1])

    return dx

def numpy_gradient(t_vec, x_vec, **kwargs):
    ord = kwargs.pop('edge_order',     2)
    return np.gradient(x_vec, t_vec, edge_order=ord, **kwargs)

def convolve(t_vec, x_vec, kernel=[1,-1], **kwargs):
    """
    https://stackoverflow.com/a/18993405
    https://stackoverflow.com/a/32544533
    """
    mod = kwargs.pop('mode', 'same')
    return np.convolve(x_vec, kernel, mode=mod) / np.convolve(t_vec, kernel, mode=mod)

def gauss(t_vec, x_vec, **kwargs):
    """
    https://stackoverflow.com/a/18993405
    https://stackoverflow.com/a/32544533
    """
    sig = kwargs.pop('sigma', 1)
    ord = kwargs.pop('order', 1)
    mod = kwargs.pop('mode', 'wrap')
    return ndimage.gaussian_filter1d(x_vec, sigma=sig, order=ord, mode=mod, **kwargs) / (t_vec[1]-t_vec[0])

def spline_diff(t_vec, x_vec, **kwargs):
    k = kwargs.pop('spline_degree',     3)                  # 3th degree spline
    s = kwargs.pop('smoothing_factor',  len(x_vec) - np.sqrt(2*len(x_vec)))   # smoothing factor
    return (UnivariateSpline(t_vec, x_vec, k=k, s=s).derivative(n=1))(t_vec)

def diff_multiple(time_step, x_vecs, method='slow', **kwargs):
    """
        time_step
            the time step between two points (maybe called tau elsewhere)
        x_vecs
            several vectors (axis 1) that vary in time on axis 0
        method                                                      (optional)
            slow                                                    (default)
                the provided function diff(t_vec, x_vec)
                order 2 on interior order 1 at boundaries
            numpy_gradient
                using numpy's gradient function
                order 2
            convolve
                optional parameters
                    kernel  = [1, -1]
                    mode    = 'same'
            gauss
                optional parameters
                    sigma   = 1
                    order   = 1
                    mode    = 'wrap'
            spline_diff
                Interpolation with a UnivariateSpline and using it's derivative
                # https://stackoverflow.com/a/36952241
                optional parameters
                    spline_degree       = 3
                    smoothing_factor    = n - np.sqrt(2*n)
    """

    if not isinstance(time_step, (np.floating, float)):
        raise TypeError('Array data type is no float!')

    if np.size(x_vecs, axis=0) == 0:
        raise ValueError('x_vecs is empty!')

    if time_step == 0:
        raise ValueError('Zero time_step not possible!')

    function_dict = {
        'slow':             diff,
        'numpy_gradient':   numpy_gradient,
        'convolve':         convolve,
        'gauss':            gauss,
        'spline_diff':      spline_diff
    }
    # Get the function from switcher dictionary
    func = function_dict.get(method, lambda: print("Invalid method"))

    dx_vecs = np.transpose(np.zeros(np.shape(x_vecs)))
    t_vec   = np.arange(0, time_step*np.size(x_vecs, axis=0), time_step)

    for k, single_time_series in enumerate(np.transpose(x_vecs)):
        dx_vecs[k] = func(t_vec, single_time_series, **kwargs)

    return np.transpose(dx_vecs)
