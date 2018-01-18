import grad_features as gf
import random_forest as rf
import numpy as np
from skimage.util.shape import view_as_windows
from scipy.ndimage import zoom
import pyximport; pyximport.install()
import nms as nms
from scipy.ndimage.filters import gaussian_filter1d
import spectrogram as sp
from skimage import filters
from scipy.io import wavfile
from skimage.util import view_as_blocks


class AudioForest:

    def __init__(self, params_):
        self.params = params_
        forest_params = rf.ForestParams(num_classes=2, trees=self.params.trees,
        depth=self.params.depth, min_cnt=self.params.min_cnt, tests=self.params.tests)
        self.forest = rf.Forest(forest_params)

    def train(self, positions, class_labels, files, durations):
        feats = []
        labs = []
        for ii, file_name in enumerate(files):

            local_feats = self.create_or_load_features(file_name)

            # convert time in file to integer
            positions_ratio = positions[ii] / durations[ii]
            train_inds = (positions_ratio*float(local_feats.shape[0])).astype('int')

            feats.append(local_feats[train_inds, :])
            labs.append(class_labels[ii])

        # flatten list of lists and set to correct output
        features = np.vstack(feats)
        labels = np.vstack(labs)
        print 'train size', features.shape
        self.forest.train(features, labels, False)

    def test(self, file_name=None, file_duration=None, audio_samples=None, sampling_rate=None):

        # compute features
        features = self.create_or_load_features(file_name, audio_samples, sampling_rate)

        # make prediction
        y_prediction = self.forest.test(features)[:, 1][:, np.newaxis]

        # smooth the output
        if self.params.smooth_op_prediction:
            y_prediction = gaussian_filter1d(y_prediction, self.params.smooth_op_prediction_sigma, axis=0)
        pos, prob = nms.nms_1d(y_prediction[:,0], self.params.nms_win_size, file_duration)

        return pos, prob, y_prediction

    def create_or_load_features(self, file_name=None, audio_samples=None, sampling_rate=None):
        """
        Does 1 of 3 possible things
        1) computes feature from audio samples directly
        2) loads feature from disk OR
        3) computes features from file name
        """

        if file_name is None:
            features = compute_features(audio_samples, sampling_rate, self.params)
        else:
            if self.params.load_features_from_file:
                features = np.load(self.params.feature_dir + file_name + '.npy')
            else:
                sampling_rate, audio_samples = wavfile.read(self.params.audio_dir + file_name + '.wav')
                features = compute_features(audio_samples, sampling_rate, self.params)
        return features

    def save_features(self, files):
        for file_name in files:
            sampling_rate, audio_samples = wavfile.read(self.params.audio_dir + file_name + '.wav')
            features = compute_features(audio_samples, sampling_rate, self.params)
            np.save(self.params.feature_dir + file_name, features)


def spatial_pool(ip, block_size):
    """
    Does sum pooling to reduce dimensionality
    """
    # make sure its evenly divisible by padding with last rows
    vert_diff = ip.shape[0]%int(block_size)
    horz_diff = ip.shape[1]%int(block_size)

    if vert_diff > 0:
        ip = np.vstack((ip, np.tile(ip[-1, :], ((block_size-vert_diff, 1)))))
    if horz_diff > 0:
        ip = np.hstack((ip, np.tile(ip[:, -1], ((block_size-horz_diff, 1))).T))

    # get block_size*block_size non-overlapping blocks
    blocks = view_as_blocks(ip, (block_size, block_size))

    # sum, could max etc.
    op = blocks.reshape(blocks.shape[0], blocks.shape[1], blocks.shape[2]*blocks.shape[3]).sum(2)

    return op


def compute_features(audio_samples, sampling_rate, params):
    """
    Computes feature vector given audio file name.
    Assumes all the spectrograms are the same size - this should be checked externally
    """

    # load audio and create spectrogram
    spectrogram = sp.gen_spectrogram(audio_samples, sampling_rate, params.fft_win_length, params.fft_overlap,
                                     crop_spec=params.crop_spec, max_freq=params.max_freq, min_freq=params.min_freq)
    spectrogram = sp.process_spectrogram(spectrogram, denoise_spec=params.denoise, mean_log_mag=params.mean_log_mag, smooth_spec=params.smooth_spec)

    # pad with dummy features at the end to take into account the size of the sliding window
    if params.feature_type == 'raw':
        spec_win = view_as_windows(spectrogram, (spectrogram.shape[0], params.window_width))[0]
        spec_win = zoom(spec_win, (1, 0.5, 0.5), order=1)
        total_win_size = spectrogram.shape[1]

    elif params.feature_type == 'grad':
        grad = np.gradient(spectrogram)
        grad_mag = np.sqrt((grad[0]**2 + grad[1]**2))
        total_win_size = spectrogram.shape[1]

        spec_win = view_as_windows(grad_mag, (grad_mag.shape[0], params.window_width))[0]
        spec_win = zoom(spec_win, (1, 0.5, 0.5), order=1)

    elif params.feature_type == 'max_freq':

        num_max_freqs = 3  # e.g. 1 means keep top 1, 2 means top 2, ...
        total_win_size = spectrogram.shape[1]
        max_freq = np.argsort(spectrogram, 0)
        max_amp = np.sort(spectrogram, 0)
        stacked = np.vstack((max_amp[-num_max_freqs:, :], max_freq[-num_max_freqs:, :]))

        spec_win = view_as_windows(stacked, (stacked.shape[0], params.window_width))[0]

    elif params.feature_type == 'hog':
        block_size = 4
        hog = gf.compute_hog(spectrogram, block_size)
        total_win_size = hog.shape[1]
        window_width_down = np.rint(params.window_width / float(block_size))

        spec_win = view_as_windows(hog, (hog.shape[0], window_width_down, hog.shape[2]))[0]

    elif params.feature_type == 'grad_pool':
        grad = np.gradient(spectrogram)
        grad_mag = np.sqrt((grad[0]**2 + grad[1]**2))

        down_sample_size = 4
        window_width_down = np.rint(params.window_width / float(down_sample_size))
        grad_mag_pool = spatial_pool(grad_mag, down_sample_size)
        total_win_size = grad_mag_pool.shape[1]

        spec_win = view_as_windows(grad_mag_pool, (grad_mag_pool.shape[0], window_width_down))[0]

    elif params.feature_type == 'raw_pool':
        down_sample_size = 4
        window_width_down = np.rint(params.window_width / float(down_sample_size))
        spec_pool = spatial_pool(spectrogram, down_sample_size)
        total_win_size = spec_pool.shape[1]

        spec_win = view_as_windows(spec_pool, (spec_pool.shape[0], window_width_down))[0]

    # pad on extra features at the end as the sliding window will mean its a different size
    features = spec_win.reshape((spec_win.shape[0], np.prod(spec_win.shape[1:])))
    features = np.vstack((features, np.tile(features[-1, :], (total_win_size - features.shape[0], 1))))
    features = features.astype(np.float32)

    return features

