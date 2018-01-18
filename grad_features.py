import numpy as np
from skimage.util import view_as_blocks, view_as_windows


def compute_hog(arr, block_size=2, block_sum=True, num_orientations=6, block_normalize=False):
    """
    Computes histogram of gradient feature for Random Forest.
    """

    # make sure the input is evenly divisible by the block size
    if block_sum:
        vert_diff = arr.shape[0]%int(block_size)
        horz_diff = arr.shape[1]%int(block_size)

        if vert_diff > 0:
            arr = np.vstack((arr, np.tile(arr[-1, :], ((vert_diff, 1)))))
        if horz_diff > 0:
            arr = np.hstack((arr, np.tile(arr[:, -1], ((horz_diff, 1))).T))

    # compute gradient magnitude and orientation
    mag, orien = gradient_mag(arr)

    # quantize orientations
    bins = np.arange(0, np.pi+np.pi/num_orientations, np.pi/num_orientations)
    orien_quantized = np.argmin(np.abs(orien[:, :, np.newaxis] - bins[np.newaxis, :]), axis=2)
    orien_quantized[orien_quantized == num_orientations] = 0

    # create histogram
    hist_of_grads = np.zeros((mag.shape[0], mag.shape[1], num_orientations))
    j, k = np.indices(mag.shape[:2])
    hist_of_grads[j, k, orien_quantized] = mag

    # add mag as extra channel
    hist_of_grads = np.dstack((hist_of_grads, mag))

    # sum over a block - note this is non-overlapping
    # note that we are assuming that hist_of_grads is evenly divisible by block_size
    if block_sum:
        blocks = view_as_blocks(hist_of_grads, (block_size, block_size, hist_of_grads.shape[2]))
        hist_of_grads = blocks.reshape(blocks.shape[0], blocks.shape[1], blocks.shape[2]*blocks.shape[3]*blocks.shape[4], blocks.shape[5]).sum(2)

    # L1 normalization
    if block_normalize:
        hist_of_grads = hist_of_grads / (hist_of_grads.sum(2) + 10e-6)[:, :, np.newaxis]

    return hist_of_grads


def gradient_mag(arr):
    """
    Computes gradient magnitude and orientation.
    """
    gx = np.empty(arr.shape, dtype=np.double)
    gx[:, 0] = 0
    gx[:, -1] = 0
    gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    gy = np.empty(arr.shape, dtype=np.double)
    gy[0, :] = 0
    gy[-1, :] = 0
    gy[1:-1, :] = arr[2:, :] - arr[:-2, :]

    mag = np.sqrt((gx**2 + gy**2))
    orien = np.arctan2(gx, gy)
    orien[orien < 0] += np.pi

    return mag, orien
