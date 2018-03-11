import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float #Do not remove this line. See http://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
def nms_1d(np.ndarray[DTYPE_t, ndim=1] src, int win_size, float file_duration):
    """1D Non maximum suppression
       src: vector of length N
    """

    cdef int max_ind = 0
    cdef int ii = 0
    cdef int ee = 0
    cdef int width = src.shape[0]-1
    cdef np.ndarray pos = np.empty(width, dtype=np.int)
    cdef int pos_cnt = 0
    while ii <= width:

        if max_ind < (ii - win_size):
            max_ind = ii - win_size

        ee = ii + win_size
        if ii + win_size >= width:
            ee = width

        while max_ind <= ee:
            if src[<unsigned int>max_ind] > src[<unsigned int>ii]:
                break
            max_ind += 1

        if max_ind > ee:
            pos[<unsigned int>pos_cnt] = ii
            pos_cnt += 1
            max_ind = ii+1
            ii += win_size

        ii += 1

    pos = pos[:pos_cnt]
    val = src[pos]

    # # remove peaks near the end
    inds = (pos + win_size) < src.shape[0]
    pos = pos[inds]
    val = val[inds]

    # set output to between 0 and 1, then put it in the correct time range
    pos = pos.astype(np.float32) / src.shape[0]
    pos = pos*file_duration

    return pos, val
