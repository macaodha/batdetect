import numpy as np
import scipy.ndimage.morphology as morph
from scipy.ndimage.filters import median_filter
import scipy.ndimage
from skimage.measure import regionprops
import spectrogram as sp
from scipy.io import wavfile

class SegmentAudio:

    def __init__(self, params_):
        self.params = params_

    def train(self, positions, class_labels, files, durations):
        # does not need to do anything
        pass

    def save_features(self, files):
        # does not need to do anything
        pass

    def test(self, file_name=None, file_duration=None, audio_samples=None, sampling_rate=None):

        sampling_rate, audio_samples = wavfile.read(self.params.audio_dir + file_name + '.wav')

        spectrogram = sp.gen_spectrogram(audio_samples, sampling_rate, self.params.fft_win_length,
                    self.params.fft_overlap, crop_spec=self.params.crop_spec, max_freq=self.params.max_freq,
                    min_freq=self.params.min_freq)
        spectrogram = sp.process_spectrogram(spectrogram, denoise_spec=self.params.denoise,
                    mean_log_mag=self.params.mean_log_mag, smooth_spec=self.params.smooth_spec)

        # compute possible call locations
        pos = compute_position_from_segment(spectrogram, file_duration, self.params)
        prob = np.ones((pos.shape[0], 1))  # no probability information
        y_prediction = np.zeros((spectrogram.shape[1], 1))  # dummy

        return pos, prob, y_prediction


def compute_position_from_segment(spec, file_duration, params):
    """
    Based on Large-scale identification of birds in audio recordings
    http://ceur-ws.org/Vol-1180/CLEF2014wn-Life-Lasseck2014.pdf
    """

    # median filter
    med_time = np.median(spec, 0)[np.newaxis, :]
    med_freq = np.median(spec, 1)[:, np.newaxis]
    med_freq_m = np.tile(med_freq, (1, spec.shape[1]))
    med_time_m = np.tile(med_time, (spec.shape[0], 1))

    # binarize
    spec_t = np.logical_and((spec > params.median_mult*med_freq_m), (spec > params.median_mult*med_time_m))

    # morphological operations
    spec_t_morph = morph.binary_closing(spec_t)
    spec_t_morph = morph.binary_dilation(spec_t_morph)
    spec_t_morph = median_filter(spec_t_morph, (2, 2))

    # connected component and filter by size
    label_im, num_labels = scipy.ndimage.label(spec_t_morph)
    sizes = scipy.ndimage.sum(spec_t_morph, label_im, range(num_labels + 1))
    mean_vals = scipy.ndimage.sum(spec, label_im, range(1, num_labels + 1))
    mask_size = sizes < params.min_region_size
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)

    # get vertical positions
    num_calls = np.unique(label_im).shape[0]-1  # no zero
    props = regionprops(label_im)
    call_pos = np.zeros(num_calls)
    for ii, pp in enumerate(props):
        call_pos[ii] = pp['bbox'][1] / float(spec.shape[1])

    # sort and convert to time as opposed to a ratio
    inds = call_pos.argsort()
    call_pos = call_pos[inds] * file_duration

    # remove overlapping calls - happens because of harmonics
    dis = np.triu(np.abs(call_pos[:, np.newaxis]-call_pos[np.newaxis, :]))
    dis = dis > params.min_overlap
    mask = np.triu(dis) + np.tril(np.ones([num_calls, num_calls]))
    valid_inds = mask.sum(0) == num_calls
    pos = call_pos[valid_inds]

    return pos
