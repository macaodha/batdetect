import numpy as np
cimport numpy as np
cimport cython

cdef inline int int_min(int a, int b): return a if a < b else b

@cython.boundscheck(False)
def nms_1d(np.ndarray src, int win_size, float file_duration):
    """1D Non maximum suppression
       src: vector of length N
    """

    cdef int src_cnt = 0
    cdef int max_ind = 0
    cdef int ii = 0
    cdef int ee = 0
    cdef int width = src.shape[0]-1
    cdef np.ndarray pos = np.empty(width, dtype=np.int)
    cdef int pos_cnt = 0
    while ii <= width:

        if max_ind < (ii - win_size):
            max_ind = ii - win_size

        ee = int_min(ii + win_size, width)

        while max_ind <= ee:
            src_cnt += 1
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

    # remove peaks near the end
    inds = (pos + win_size) < src.shape[0]
    pos = pos[inds]
    val = val[inds]

    # set output to between 0 and 1, then put it in the correct time range
    pos = pos.astype(np.float) / src.shape[0]
    pos = pos*file_duration

    return pos, val[..., np.newaxis]
