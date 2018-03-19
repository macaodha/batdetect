from __future__ import print_function
import numpy as np

def nms_1d(src, win_size, file_duration):
    """1D Non maximum suppression
       src: vector of length N
    """

    pos = []
    src_cnt = 0
    max_ind = 0
    ii = 0
    ee = 0
    width = src.shape[0]-1
    while ii <= width:

        if max_ind < (ii - win_size):
            max_ind = ii - win_size

        ee = np.minimum(ii + win_size, width)

        while max_ind <= ee:
            src_cnt += 1
            if src[int(max_ind)] > src[int(ii)]:
                break
            max_ind += 1

        if max_ind > ee:
            pos.append(ii)
            max_ind = ii+1
            ii += win_size

        ii += 1

    pos = np.asarray(pos).astype(np.int)
    val = src[pos]

    # remove peaks near the end
    inds = (pos + win_size) < src.shape[0]
    pos = pos[inds]
    val = val[inds]

    # set output to between 0 and 1, then put it in the correct time range
    pos = pos / float(src.shape[0])
    pos = pos*file_duration

    return pos, val


def test_nms():
    import matplotlib.pyplot as plt
    import numpy as np
    #import pyximport; pyximport.install(reload_support=True)
    import nms as nms_fast

    y = np.sin(np.arange(1000)/100.0*np.pi)
    y = y + np.random.random(y.shape)*0.5
    win_size = int(0.1*y.shape[0]/2.0)

    pos, prob = nms_1d(y, win_size, y.shape[0])
    pos_f, prob_f = nms_fast.nms_1d(y, win_size, y.shape[0])

    print('diff between implementations =', 1-np.isclose(prob_f, prob).mean())
    print('diff between implementations =', 1-np.isclose(pos_f, pos).mean())

    plt.close('all')
    plt.plot(y)
    plt.plot((pos).astype('int'), prob, 'ro', ms=10)
    plt.plot((pos_f).astype('int'), prob, 'bo')  # shift so we can see them
    plt.show()
