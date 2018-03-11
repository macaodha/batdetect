"""
This file contains code copied from skimage package. 
Specifically, this file is a standalone implementation of 
skimage.filters's "gaussian" function.
The "image_as_float" and "guess_spatial_dimensions" functions
were also copied to as dependencies of "gaussian" function.
"""
from __future__ import division
import numbers
import collections as coll
import numpy as np
from scipy import ndimage as ndi

__all__ = ['gaussian']

dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.int64: (-2**63, 2**63 - 1),
               np.uint64: (0, 2**64 - 1),
               np.int32: (-2**31, 2**31 - 1),
               np.uint32: (0, 2**32 - 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}

integer_types = (np.uint8, np.uint16, np.int8, np.int16)

_supported_types = (np.bool_, np.bool8,
                    np.uint8, np.uint16, np.uint32, np.uint64,
                    np.int8, np.int16, np.int32, np.int64,
                    np.float32, np.float64)

dtype_range[np.float16] = (-1, 1)
_supported_types += (np.float16, )

def warn(msg):
    print(msg)


def img_as_float(image):
    dtype=np.float32
    force_copy = False

    """
    Convert an image to the requested data-type.
    Warnings are issued in case of precision loss, or when negative values
    are clipped during conversion to unsigned integer types (sign loss).
    Floating point values are expected to be normalized and will be clipped
    to the range [0.0, 1.0] or [-1.0, 1.0] when converting to unsigned or
    signed integers respectively.
    Numbers are not shifted to the negative side when converting from
    unsigned to signed integer types. Negative values will be clipped when
    converting to unsigned integers.
    Parameters
    ----------
    image : ndarray
        Input image.
    dtype : dtype
        Target data-type.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    uniform : bool, optional
        Uniformly quantize the floating point range to the integer range.
        By default (uniform=False) floating point values are scaled and
        rounded to the nearest integers, which minimizes back and forth
        conversion errors.
    References
    ----------
    .. [1] DirectX data conversion rules.
           http://msdn.microsoft.com/en-us/library/windows/desktop/dd607323%28v=vs.85%29.aspx
    .. [2] Data Conversions. In "OpenGL ES 2.0 Specification v2.0.25",
           pp 7-8. Khronos Group, 2010.
    .. [3] Proper treatment of pixels as integers. A.W. Paeth.
           In "Graphics Gems I", pp 249-256. Morgan Kaufmann, 1990.
    .. [4] Dirty Pixels. J. Blinn. In "Jim Blinn's corner: Dirty Pixels",
           pp 47-57. Morgan Kaufmann, 1998.
    """
    image = np.asarray(image)
    dtypeobj = np.dtype(dtype)
    dtypeobj_in = image.dtype
    dtype = dtypeobj.type
    dtype_in = dtypeobj_in.type

    if dtype_in == dtype:
        if force_copy:
            image = image.copy()
        return image

    if not (dtype_in in _supported_types and dtype in _supported_types):
        raise ValueError("can not convert %s to %s." % (dtypeobj_in, dtypeobj))

    def sign_loss():
        warn("Possible sign loss when converting negative image of type "
             "%s to positive image of type %s." % (dtypeobj_in, dtypeobj))

    def prec_loss():
        warn("Possible precision loss when converting from "
             "%s to %s" % (dtypeobj_in, dtypeobj))

    def _dtype(itemsize, *dtypes):
        # Return first of `dtypes` with itemsize greater than `itemsize`
        return next(dt for dt in dtypes if itemsize < np.dtype(dt).itemsize)

    def _dtype2(kind, bits, itemsize=1):
        # Return dtype of `kind` that can store a `bits` wide unsigned int
        def compare(x, y, kind='u'):
            if kind == 'u':
                return x <= y
            else:
                return x < y

        s = next(i for i in (itemsize, ) + (2, 4, 8) if compare(bits, i * 8,
                                                                kind=kind))
        return np.dtype(kind + str(s))


    def _scale(a, n, m, copy=True):
        # Scale unsigned/positive integers from n to m bits
        # Numbers can be represented exactly only if m is a multiple of n
        # Output array is of same kind as input.
        kind = a.dtype.kind
        if n > m and a.max() < 2 ** m:
            mnew = int(np.ceil(m / 2) * 2)
            if mnew > m:
                dtype = "int%s" % mnew
            else:
                dtype = "uint%s" % mnew
            n = int(np.ceil(n / 2) * 2)
            msg = ("Downcasting %s to %s without scaling because max "
                   "value %s fits in %s" % (a.dtype, dtype, a.max(), dtype))
            warn(msg)
            return a.astype(_dtype2(kind, m))
        elif n == m:
            return a.copy() if copy else a
        elif n > m:
            # downscale with precision loss
            prec_loss()
            if copy:
                b = np.empty(a.shape, _dtype2(kind, m))
                np.floor_divide(a, 2**(n - m), out=b, dtype=a.dtype,
                                casting='unsafe')
                return b
            else:
                a //= 2**(n - m)
                return a
        elif m % n == 0:
            # exact upscale to a multiple of n bits
            if copy:
                b = np.empty(a.shape, _dtype2(kind, m))
                np.multiply(a, (2**m - 1) // (2**n - 1), out=b, dtype=b.dtype)
                return b
            else:
                a = np.array(a, _dtype2(kind, m, a.dtype.itemsize), copy=False)
                a *= (2**m - 1) // (2**n - 1)
                return a
        else:
            # upscale to a multiple of n bits,
            # then downscale with precision loss
            prec_loss()
            o = (m // n + 1) * n
            if copy:
                b = np.empty(a.shape, _dtype2(kind, o))
                np.multiply(a, (2**o - 1) // (2**n - 1), out=b, dtype=b.dtype)
                b //= 2**(o - m)
                return b
            else:
                a = np.array(a, _dtype2(kind, o, a.dtype.itemsize), copy=False)
                a *= (2**o - 1) // (2**n - 1)
                a //= 2**(o - m)
                return a

    kind = dtypeobj.kind
    kind_in = dtypeobj_in.kind
    itemsize = dtypeobj.itemsize
    itemsize_in = dtypeobj_in.itemsize

    if kind == 'b':
        # to binary image
        if kind_in in "fi":
            sign_loss()
        prec_loss()
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    if kind_in == 'b':
        # from binary image, to float and to integer
        result = image.astype(dtype)
        if kind != 'f':
            result *= dtype(dtype_range[dtype][1])
        return result

    if kind in 'ui':
        imin = np.iinfo(dtype).min
        imax = np.iinfo(dtype).max
    if kind_in in 'ui':
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max

    if kind_in == 'f':
        if np.min(image) < -1.0 or np.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        if kind == 'f':
            # floating point -> floating point
            if itemsize_in > itemsize:
                prec_loss()
            return image.astype(dtype)

        # floating point -> integer
        prec_loss()
        # use float type that can represent output integer type
        image = np.array(image, _dtype(itemsize, dtype_in,
                                       np.float32, np.float64))
        if not uniform:
            if kind == 'u':
                image *= imax
            else:
                image *= imax - imin
                image -= 1.0
                image /= 2.0
            np.rint(image, out=image)
            np.clip(image, imin, imax, out=image)
        elif kind == 'u':
            image *= imax + 1
            np.clip(image, 0, imax, out=image)
        else:
            image *= (imax - imin + 1.0) / 2.0
            np.floor(image, out=image)
            np.clip(image, imin, imax, out=image)
        return image.astype(dtype)

    if kind == 'f':
        # integer -> floating point
        if itemsize_in >= itemsize:
            prec_loss()
        # use float type that can exactly represent input integers
        image = np.array(image, _dtype(itemsize_in, dtype,
                                       np.float32, np.float64))
        if kind_in == 'u':
            image /= imax_in
            # DirectX uses this conversion also for signed ints
            #if imin_in:
            #    np.maximum(image, -1.0, out=image)
        else:
            image *= 2.0
            image += 1.0
            image /= imax_in - imin_in
        return image.astype(dtype)

    if kind_in == 'u':
        if kind == 'i':
            # unsigned integer -> signed integer
            image = _scale(image, 8 * itemsize_in, 8 * itemsize - 1)
            return image.view(dtype)
        else:
            # unsigned integer -> unsigned integer
            return _scale(image, 8 * itemsize_in, 8 * itemsize)

    if kind == 'u':
        # signed integer -> unsigned integer
        sign_loss()
        image = _scale(image, 8 * itemsize_in - 1, 8 * itemsize)
        result = np.empty(image.shape, dtype)
        np.maximum(image, 0, out=result, dtype=image.dtype, casting='unsafe')
        return result

    # signed integer -> signed integer
    if itemsize_in > itemsize:
        return _scale(image, 8 * itemsize_in - 1, 8 * itemsize - 1)
    image = image.astype(_dtype2('i', itemsize * 8))
    image -= imin_in
    image = _scale(image, 8 * itemsize_in, 8 * itemsize, copy=False)
    image += imin

    return image.astype(dtype)



def guess_spatial_dimensions(image):
    """Make an educated guess about whether an image has a channels dimension.
    Parameters
    ----------
    image : ndarray
        The input image.
    Returns
    -------
    spatial_dims : int or None
        The number of spatial dimensions of `image`. If ambiguous, the value
        is ``None``.
    Raises
    ------
    ValueError
        If the image array has less than two or more than four dimensions.
    """
    if image.ndim == 2:
        return 2
    if image.ndim == 3 and image.shape[-1] != 3:
        return 3
    if image.ndim == 3 and image.shape[-1] == 3:
        return None
    if image.ndim == 4 and image.shape[-1] == 3:
        return 3
    else:
        raise ValueError("Expected 2D, 3D, or 4D array, got %iD." % image.ndim)


def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
             multichannel=None):
    """Multi-dimensional Gaussian filter

    Parameters
    ----------
    image : array-like
        Input image (grayscale or color) to filter.
    sigma : scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    output : array, optional
        The ``output`` parameter passes an array in which to store the
        filter output.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'nearest'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0
    multichannel : bool, optional (default: None)
        Whether the last axis of the image is to be interpreted as multiple
        channels. If True, each channel is filtered separately (channels are
        not mixed together). Only 3 channels are supported. If `None`,
        the function will attempt to guess this, and raise a warning if
        ambiguous, when the array has shape (M, N, 3).

    Returns
    -------
    filtered_image : ndarray
        the filtered array

    Notes
    -----
    This function is a wrapper around :func:`scipy.ndi.gaussian_filter`.

    Integer arrays are converted to float.

    The multi-dimensional filter is implemented as a sequence of
    one-dimensional convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.

    Examples
    --------

    >>> a = np.zeros((3, 3))
    >>> a[1, 1] = 1
    >>> a
    array([[ 0.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  0.]])
    >>> gaussian(a, sigma=0.4)  # mild smoothing
    array([[ 0.00163116,  0.03712502,  0.00163116],
           [ 0.03712502,  0.84496158,  0.03712502],
           [ 0.00163116,  0.03712502,  0.00163116]])
    >>> gaussian(a, sigma=1)  # more smooting
    array([[ 0.05855018,  0.09653293,  0.05855018],
           [ 0.09653293,  0.15915589,  0.09653293],
           [ 0.05855018,  0.09653293,  0.05855018]])
    >>> # Several modes are possible for handling boundaries
    >>> gaussian(a, sigma=1, mode='reflect')
    array([[ 0.08767308,  0.12075024,  0.08767308],
           [ 0.12075024,  0.16630671,  0.12075024],
           [ 0.08767308,  0.12075024,  0.08767308]])
    >>> # For RGB images, each is filtered separately
    >>> from skimage.data import astronaut
    >>> image = astronaut()
    >>> filtered_img = gaussian(image, sigma=1, multichannel=True)

    """

    spatial_dims = guess_spatial_dimensions(image)
    if spatial_dims is None and multichannel is None:
        msg = ("Images with dimensions (M, N, 3) are interpreted as 2D+RGB "
               "by default. Use `multichannel=False` to interpret as "
               "3D image with last dimension of length 3.")
        warn(RuntimeWarning(msg))
        multichannel = True
    if np.any(np.asarray(sigma) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")
    if multichannel:
        # do not filter across channels
        if not isinstance(sigma, coll.Iterable):
            sigma = [sigma] * (image.ndim - 1)
        if len(sigma) != image.ndim:
            sigma = np.concatenate((np.asarray(sigma), [0]))
    #image = img_as_float(image)
    return ndi.gaussian_filter(image, sigma, mode=mode, cval=cval)
