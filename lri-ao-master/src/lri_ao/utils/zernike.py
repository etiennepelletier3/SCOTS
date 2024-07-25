import numpy as np
from scipy.special import factorial as fact
import matplotlib.pyplot as plt
from .nollIndex import NollIndex2zernike as n2z


def kronekerDelta(a, b):
    if np.isclose(a, b):
        return 1
    else:
        return 0


def zernikeR(n, m, radius, circle=True):
    # return R_n^m(r) using :Adaptative optics for astronomy p.91 (2)
    if (n - m) % 2 != 0:
        return 0
    else:
        m = abs(m)
        s = []
        for k in np.arange(int((n - m) / 2 + 1)):
            s.append(
                (
                    ((-1) ** k * fact(n - k))
                    / (fact(k) * fact(((n + m) / 2) - k) * fact(((n - m) / 2) - k))
                )
                * radius ** (n - 2 * k)
            )
        if circle is True:
            return np.choose(radius > 1, [np.sum(s, axis=0), 0])
        else:
            return np.sum(s, axis=0)


def zernikeN(n, m):
    """Normalisation constant for Zernike polynomials,
    Adaptative optics for astronomy p.91 (1)"""
    return (
        np.sqrt(n + 1)
        * np.sqrt(2)
        / (np.sqrt(1 + kronekerDelta(m, 0)) * np.sqrt(np.pi))
    )


def listReturn(arg):
    """If the argument is an array, returns an array,
    else it returns a 0 sized array"""
    argArray = np.array(arg)
    try:
        # Fails gracefully
        argArray[0]
    except IndexError or TypeError:
        # Fails gracefully
        argArray = np.array((arg,))
    return argArray


def zernike(indexList, radius, angle, weight=None, circle=True):
    """From an indexList [[n1,m1],[n2,m2]...] gives the sum of the
    zernike coefficients according to the weight (normalization) given.
    Also accepts the form: indexList = (n,m) and weight = integer
    for a single zernike coefficient"""

    indexArray = np.array(indexList)

    # assumes its multiple zernike coefficient
    try:
        indexArray.shape[1]
        if weight is None:
            weight = np.ones(indexArray.shape[0])
        else:
            weight = listReturn(weight)
        nArray = indexArray[:, 0]
        mArray = indexArray[:, 1]

    # fails gracefully if it isn't the case
    except IndexError:
        if weight is None:
            weight = np.ones(1)
        nArray = (indexArray[0],)
        mArray = (indexArray[1],)
        weight = (weight,)

    # sums the different zernike coefficients using the zernikeR function
    wavefrontsol = 0
    for zern_index in range(len(nArray)):
        if mArray[zern_index] >= 0:
            wavefrontsol += (
                weight[zern_index]
                * zernikeN(nArray[zern_index], mArray[zern_index])
                * zernikeR(nArray[zern_index], mArray[zern_index], radius, circle)
                * np.cos(mArray[zern_index] * angle)
            )
        else:
            wavefrontsol += (
                weight[zern_index]
                * zernikeN(nArray[zern_index], mArray[zern_index])
                * zernikeR(nArray[zern_index], abs(mArray[zern_index]), radius, circle)
                * np.sin(abs(mArray[zern_index]) * angle)
            )
    return wavefrontsol


def gradZernike(indexList, radius, angle, weight=None):
    """Gives out the gradient of the specified Zernike"""
    return np.choose(
        radius >= 1,
        [np.gradient(zernike(indexList, radius, angle, weight, circle=False)), 0],
    )


def noll_gradZernike(indexList, radius, angle, weight=None, circle=True):
    if circle is True:
        return np.choose(
            radius >= 1,
            [
                np.gradient(
                    noll_zernike(indexList, radius, angle, weight, circle=False)
                ),
                0,
            ],
        )
    else:
        return np.gradient(noll_zernike(indexList, radius, angle, weight, circle=False))


def noll2zernikeIndex(indexList):
    """, we use the Noll index to return the polynomials."""
    nollArray = listReturn(indexList)
    zernikeArray = []
    for noll_index in nollArray:
        zernikeArray.append(n2z[str(noll_index)])
    return zernikeArray


def noll_zernike(indexList, radius, angle, weight=None, circle=True):
    """Uses Noll index list to return the Zernike"""
    return zernike(
        noll2zernikeIndex(indexList), radius, angle, weight=weight, circle=circle
    )


def fourier(indexList, x, y, weight=None, circle=True):
    """Accepts indexList in the form [i1,i2,i3,i4,...], returns fourier modes
    on an unit circle.
    Frequency given by the index in groups of 4.
    index//4 -> frequency (multiples).
    index%4 -> function: (0,1,2,3) -> (sin(x),cos(x),sin(y),cos(y))"""

    indexArray = np.array(indexList)

    # assumes its multiple zernike coefficient
    try:
        indexArray.shape[1]
        if weight is None:
            weight = np.ones(indexArray.shape[0])
        else:
            weight = listReturn(weight)

    # fails gracefully if it isn't the case
    except IndexError:
        if weight is None:
            weight = np.ones(1)
        weight = (weight,)

    # sums the different zernike coefficients using the zernikeR function
    wavefrontsol = 0
    for index in range(len(indexArray)):
        # sine contribution
        wavefrontsol += (1 - indexArray[index] % 2) * np.sin(
            (
                ((1 - ((indexArray[index]) % 4 < 2)) * y)
                + ((((indexArray[index]) % 4 < 2)) * x)
            )
            * (indexArray[index] // 4 + 1)
            * np.pi
        )
        # cosine_contribution
        wavefrontsol += (indexArray[index] % 2) * np.cos(
            (
                ((1 - ((indexArray[index]) % 4 < 2)) * y)
                + ((((indexArray[index]) % 4 < 2)) * x)
            )
            * (indexArray[index] // 4 + 1)
            * np.pi
        )
        # weight
        wavefrontsol *= weight[index]
    return wavefrontsol
