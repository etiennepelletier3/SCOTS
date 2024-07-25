import numpy as np
from scipy.ndimage import zoom as ndimage_zoom
from cv2 import moments
from warnings import warn


def cog(input_array):
    """Simple COG measurement using cv2"""

    a_min = input_array.min()
    a_max = input_array.max()
    # ret, input_array_thr = threshold(
    #     input_array,
    #     a_min + (a_max - a_min) / 3,
    #     a_max,
    #     THRESH_TOZERO,
    # )
    input_array_fn = np.copy(input_array)
    input_array_fn[
        input_array <= input_array[input_array < a_min + (a_max - a_min / 3)].mean()
    ] = 0
    M = moments(input_array_fn)
    if np.isclose(M["m00"], 0):
        warn('Could not compute centroid since moment "m00" is 0')
        return np.array((0, 0)).reshape(2, 1)

    return np.array([M["m01"] / M["m00"], M["m10"] / M["m00"]]).reshape(2, 1)


def zoom(input, zoom, **kwargs):
    return ndimage_zoom(input, zoom, **kwargs)


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get("padder", 0)
    vector[: pad_width[0]] = pad_value
    vector[-pad_width[1] :] = pad_value
    return vector


def zoom_array(
    inArray, finalShape, sameSum=False, zoomFunction=ndimage_zoom, **zoomKwargs
):
    """

    Normally, one can use scipy.ndimage.zoom to do array/image rescaling.
    However, scipy.ndimage.zoom does not coarsegrain images well. It basically
    takes nearest neighbor, rather than averaging all the pixels, when
    coarsegraining arrays. This increases noise. Photoshop doesn't do that, and
    performs some smart interpolation-averaging instead.

    If you were to coarsegrain an array by an integer factor, e.g. 100x100 ->
    25x25, you just need to do block-averaging, that's easy, and it reduces
    noise. But what if you want to coarsegrain 100x100 -> 30x30?

    Then my friend you are in trouble. But this function will help you. This
    function will blow up your 100x100 array to a 120x120 array using
    scipy.ndimage zoom Then it will coarsegrain a 120x120 array by
    block-averaging in 4x4 chunks.

    It will do it independently for each dimension, so if you want a 100x100
    array to become a 60x120 array, it will blow up the first and the second
    dimension to 120, and then block-average only the first dimension.

    Parameters
    ----------

    inArray: n-dimensional numpy array (1D also works)
    finalShape: resulting shape of an array
    sameSum: bool, preserve a sum of the array, rather than values.
             by default, values are preserved
    zoomFunction: by default, scipy.ndimage.zoom. You can plug your own.
    zoomKwargs:  a dict of options to pass to zoomFunction.
    """
    inArray = np.asarray(inArray, dtype=np.double)
    inShape = inArray.shape
    assert len(inShape) == len(finalShape)
    mults = []  # multipliers for the final coarsegraining
    for i in range(len(inShape)):
        if finalShape[i] < inShape[i]:
            mults.append(int(np.ceil(inShape[i] / finalShape[i])))
        else:
            mults.append(1)
    # shape to which to blow up
    tempShape = tuple([i * j for i, j in zip(finalShape, mults)])

    # stupid zoom doesn't accept the final shape. Carefully crafting the
    # multipliers to make sure that it will work.
    zoomMultipliers = np.array(tempShape) / np.array(inShape) + 0.0000001
    assert zoomMultipliers.min() >= 1

    # applying scipy.ndimage.zoom
    rescaled = zoomFunction(inArray, zoomMultipliers, **zoomKwargs)

    for ind, mult in enumerate(mults):
        if mult != 1:
            sh = list(rescaled.shape)
            assert sh[ind] % mult == 0
            newshape = sh[:ind] + [sh[ind] // mult, mult] + sh[ind + 1 :]
            rescaled.shape = newshape
            rescaled = np.mean(rescaled, axis=ind + 1)
    assert rescaled.shape == tuple(finalShape.astype(int))

    if sameSum:
        extraSize = np.prod(finalShape) / np.prod(inShape)
        rescaled /= extraSize
    return rescaled


def fixed_zoom_array(array, new_shape):
    size = array.shape[0]
    zoomed_shape = new_shape
    if zoomed_shape[0] / size == 1:
        answer = array
    if zoomed_shape[0] / size > 1:
        answer = zoom_array(array, zoomed_shape)[
            int((zoomed_shape[0] / 2) - (size / 2)) : int(
                (zoomed_shape[0] / 2) + (size / 2)
            ),
            int((zoomed_shape[0] / 2) - (size / 2)) : int(
                (zoomed_shape[0] / 2) + (size / 2)
            ),
        ]
    else:
        zoom = zoom_array(array, zoomed_shape)
        padded = np.lib.pad(
            zoom, int(np.ceil((array.shape[0] - zoom.shape[0]) / 2)), "constant"
        )
        if padded.shape[0] == array.shape[0]:
            answer = padded
        elif (padded[1:, 1:]).shape[0] == array.shape[0]:
            answer = padded[1:, 1:]
    return answer
