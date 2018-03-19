import numpy as np
from numpy.lib.stride_tricks import as_strided


def aligned_malloc(shape, dtype, alignment=16):
    """allocates numpy.array of specified shape, dtype
    and memory alignment such that array.ctypes.data
    is an aligned memory pointer
    shape is numpy.array shape
    dtype is numpy.array element type
    alignment is required memory alignment in bytes
    """
    itemsize = np.dtype(dtype).itemsize
    extra = alignment // itemsize
    size = np.prod(shape)
    buf = np.empty(size + extra, dtype=dtype)
    ofs = (-buf.ctypes.data % alignment) // itemsize
    aa = buf[ofs:ofs+size].reshape(shape)
    assert (aa.ctypes.data % alignment) == 0
    assert (aa.flags['C_CONTIGUOUS']) == True
    return aa


def view_as_windows(arr_in, window_shape, step=1):
    """ Taken from skimage.util.shape.py
    Rolling window view of the input n-dimensional array.

    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).

    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    window_shape : tuple
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
    step : int, optional
        Number of elements to skip when moving the window forward (by
        default, move forward by one). The value must be equal or larger
        than one.

    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.   If `arr_in` is
        non-contiguous, a copy is made.
    """

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    # -- build rolling window view
    arr_in = np.ascontiguousarray(arr_in)

    new_shape = tuple((arr_shape - window_shape) // step + 1) + \
        tuple(window_shape)

    arr_strides = np.array(arr_in.strides)
    new_strides = np.concatenate((arr_strides * step, arr_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out


def corr2d(ip, filters, bias):
    """performs 2D correlation on 3D input matrix with depth D, with N filters
    does matrix multiplication method - will use a lot of memory for large
    inputs. see here for more details:
    https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
    ip is DxHxW
    filters is NxDxFhxFw, where Fh==Fw
    op is NxHxW
    """

    # reshape filters, can do this outside as it only needs to be done once
    filters_re = filters.reshape((filters.shape[0], np.prod(filters.shape[1:])))

    # produce views of the input
    op = view_as_windows(ip, filters.shape[1:])
    op_height, op_width = op.shape[1:3]

    # reshape to 2D matrix and correlate with filters
    op = op.reshape((np.prod(op.shape[:3]), np.prod(op.shape[3:])))
    op = np.dot(filters_re, op.T)

    # reshape back to the correct op size
    op = op.reshape((filters.shape[0], op_height, op_width))

    # add bias term
    op += bias[..., np.newaxis, np.newaxis]

    # non linearity - ReLu
    op.clip(min=0, out=op)

    return op


def max_pool(ip):
    """does a 2x2 max pool, crops off ends if not divisible by 2
    ip is DxHxW
    op is DxH/2xW/2
    """

    height = ip.shape[1] - ip.shape[1]%2
    width = ip.shape[2] - ip.shape[2]%2
    h_max = np.maximum(ip[:,:height:2,:], ip[:,1:height:2,:])
    op = np.maximum(h_max[:,:,:width:2], h_max[:,:,1:width:2])
    return op


def fully_connected_as_corr(ip, filters, bias):
    """turns a conv ouput to fully connected layer into a correlation by sliding
    it across the horizontal direction. this only needs to happen in 1D as the
    nuerons see the same size as the input
    ip is DxHxW
    filters is 2D - (DxHxW)x(num_neurons)
    op is Wxnum_neurons
    """

    # create DxHxsliding_width views of input - similar to corr2d
    sliding_width = filters.shape[0] // np.prod(ip.shape[:2])
    op = view_as_windows(ip, (ip.shape[0],ip.shape[1],sliding_width))
    op = op.reshape((np.prod(op.shape[:3]), np.prod(op.shape[3:])))

    # perform correlation view matrix multiplication
    op = np.dot(op, filters)

    # add bias term
    op += bias[np.newaxis, :]

    # non linearity - ReLu
    op.clip(min=0, out=op)

    # pad with zeros at end so thats it the same width as input
    op = np.vstack((op, np.zeros((ip.shape[2]-op.shape[0], op.shape[1]), dtype=np.float32)))

    return op

