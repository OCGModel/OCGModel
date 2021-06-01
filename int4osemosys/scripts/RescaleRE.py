import numpy as np
import scipy.interpolate  # Interpolate input parameters


def _mapping(xx, yy, x):
    fun = scipy.interpolate.interp1d(xx, yy)
    res = fun(x)

    ii = x <= xx[0]
    res[ii] = yy[0]

    jj = x >= xx[-1]
    res[jj] = yy[-1]

    return res


def _bisect(fun, xi, xj, min_step):
    fxi = fun(xi)
    fxj = fun(xj)

    # all values were mapped and still the set FLH where not achieved
    if fxi * fxj > 0:
        raise ValueError("Error in rescaling: FLH not reachable!")
    else:
        while True:
            xk = (xi + xj) / 2  # take the medium point.
            fxk = fun(xk)

            ## All indexes where the
            if np.sign(fxi) == np.sign(fxk):
                xi = xk
                fxi = fxk

            else:
                xj = xk
                fxj = fxk

            # test if i reached the min step for all end
            if np.absolute(xi - xj) <= min_step:
                break
        return xk


def rescale_re_flh(ts_filename, flh, save=False):
    """
    Rescale VREs availability to meet a given full load hours.

    Rescale the availability values to meet the given full load hours **flh**, by modifying intermediate values while
    keeping minimum and maximum values fixed.

    Args:
        ts_filename (str): Path to timeseries text file. The availbility timeseries must have 8760 entries,
                           i.e. one for each your of the year.
        flh (float): The full load hours the availability time series must be rescaled to.
        save (bool, optional): Indicator if the new availbility timeseries should be save. Defaults to False.
                               If True, the new time series is saved at the save path of the **ts_filename** with
                               "_**flh**FLH" added to the name.

    Returns:
        list: List with len 8760 with the rescaled availability factors.

    """
    # load x from file
    x = np.loadtxt(ts_filename)
    if len(x) != 8760:
        raise ValueError("Availability curve does not have 8760 entries!")

    # max availability
    maxx = np.max(x)

    # Mapping fun. calculate new time series given a scale value a.
    mapp = lambda a: _mapping([0, maxx / 2, maxx], [0, a, 1], x)

    # Fun to calculate the FLH missing considering if "a" was assumed
    flh_missing = lambda a: np.mean(mapp(a)) * 8760 - flh

    # iterative calculates the value, starting at the limits of the interval:
    a = _bisect(flh_missing, 0, 1, 0.001)

    ret = mapp(a)
    # save rescaled timeseries
    if save:
        save_filename = ts_filename.split(".")
        save_filename[0] = save_filename[0] + "_%sFLH" % flh
        save_filename = ".".join(save_filename)
        with open(save_filename, "wb") as f:
            np.savetxt(f, ret, fmt="%.6f")

    return list(ret)


if __name__ == '__main__':
    pass
