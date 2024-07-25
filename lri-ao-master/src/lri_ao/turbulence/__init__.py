import numpy as np


def __strided_method(ar):
    a = np.concatenate((ar, ar[:-1]))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L - 1 :], (L, L), (-n, n)).copy()


def __row1fork(ar):
    return (ar.T - ar[:, 0]).T


def structure(arr):
    """From phase array computes structure function using strided views"""
    locind = np.indices(arr.shape)
    locindx = np.ravel(locind[0])
    locindy = np.ravel(locind[1])
    ravelarr = np.ravel(arr)
    mask = ~np.isnan(ravelarr)
    locindx = locindx[mask]
    locindy = locindy[mask]
    ravelarr = ravelarr[mask]

    matv = __strided_method(ravelarr)
    matx = __strided_method(locindx)
    maty = __strided_method(locindy)

    mask = np.tri(matv.shape[0], k=-1)
    mask[0] = 1
    mask[:,0] = 0
    mask = np.nonzero(mask)

    val = np.abs(__row1fork(matv)[mask]) ** 2
    matx = __row1fork(matx)[mask]
    maty = __row1fork(maty)[mask]
    d = np.sqrt(matx ** 2 + maty ** 2)

    unq, unq_idx, unq_inv, unq_cnt = np.unique(
        d, return_index=True, return_inverse=True, return_counts=True
    )
    unique_dist = d[unq_idx]
    unique_mean = np.bincount(unq_inv, val) / unq_cnt
    return unique_dist, unique_mean
