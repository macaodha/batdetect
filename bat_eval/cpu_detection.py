from __future__ import print_function
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter1d
import json
import time

from spectrogram import Spectrogram
import cnn_helpers as ch

import warnings
warnings.simplefilter("ignore", UserWarning)

try:
    import nms as nms
except ImportError as e:
    print("Import Error: {0}".format(e))
    print('please compile fast nms by running:')
    print('python setup.py build_ext --inplace')
    print('using slow nms in the meantime.')
    import nms_slow as nms


class CPUDetector:

    def __init__(self, weight_file, params_file):
        """Performs detection on an audio file.
        The structure of the network is hard coded to a network with 2
        convolution layers with pooling, 1 or 2 fully connected layers, and a
        final softmax layer.

        weight_file is the path to the numpy weights to the network
        params_file is the path to the network parameters
        """

        self.weights = np.load(weight_file, encoding='latin1')
        if not all([weight.dtype==np.float32 for weight in self.weights]):
            for i in range(self.weights.shape[0]):
                self.weights[i] = self.weights[i].astype(np.float32)

        with open(params_file) as fp:
            params = json.load(fp)

        self.chunk_size = 4.0  # seconds
        self.win_size = params['win_size']
        self.max_freq = params['max_freq']
        self.min_freq = params['min_freq']
        self.slice_scale = params['slice_scale']
        self.overlap = params['overlap']
        self.crop_spec = params['crop_spec']
        self.denoise = params['denoise']
        self.smooth_spec = params['smooth_spec']
        self.nms_win_size = int(params['nms_win_size'])
        self.smooth_op_prediction_sigma = params['smooth_op_prediction_sigma']
        self.sp = Spectrogram()


    def run_detection(self, spec, chunk_duration, detection_thresh, low_res=True):
        """audio is the raw samples should be 1D vector
        sampling_rate should be divided by 10 if the recordings are not time
        expanded
        """

        # run the cnn - low_res will be faster but less accurate
        if low_res:
            prob = self.eval_network(spec)
            scale_fact = 8.0
        else:
            prob_1 = self.eval_network(spec)
            prob_2 = self.eval_network(spec[:, 2:])

            prob = np.zeros(prob_1.shape[0]*2, dtype=np.float32)
            prob[0::2] = prob_1
            prob[1::2] = prob_2
            scale_fact = 4.0

        f_size = self.smooth_op_prediction_sigma / scale_fact
        nms_win = np.round(self.nms_win_size / scale_fact)

        # smooth the outputs - this might not be necessary
        prob = gaussian_filter1d(prob, f_size)

        # perform non maximum suppression
        call_time, call_prob = nms.nms_1d(prob, nms_win, chunk_duration)

        # remove detections below threshold
        if call_prob.shape[0] > 0:
            inds = (call_prob >= detection_thresh)
            call_prob = call_prob[inds]
            call_time = call_time[inds]

        return call_time, call_prob


    def create_spec(self, audio, sampling_rate):
        """Creates spectrogram (returned numpy array has correct memory alignement)
        """
        hspec = self.sp.gen_spectrogram(audio, sampling_rate, self.slice_scale,
                                    self.overlap, crop_spec=self.crop_spec,
                                    max_freq=self.max_freq, min_freq=self.min_freq)
        hspec = self.sp.process_spectrogram(hspec, denoise_spec=self.denoise,
                                    smooth_spec=self.smooth_spec)
        nsize = (np.ceil(hspec.shape[0]/2.0).astype(int), np.ceil(hspec.shape[1]/2.0).astype(int))
        spec = ch.aligned_malloc(nsize, np.float32)

        zoom(hspec, 0.5, output=spec, order=1)
        return spec


    def eval_network(self, ip):
        """runs the cnn - either the 1 or 2 fully connected versions
        """

        if self.weights.shape[0] == 8:
            prob = self.eval_network_1_dense(ip)
        elif self.weights.shape[0] == 10:
            prob = self.eval_network_2_dense(ip)
        return prob


    def eval_network_1_dense(self, ip):
        """cnn with 1 dense layer at end
        """

        # Conv Layer 1
        conv1 = ch.corr2d(ip[np.newaxis,:,:], self.weights[0], self.weights[1])
        pool1 = ch.max_pool(conv1)

        # Conv Layer 2
        conv2 = ch.corr2d(pool1, self.weights[2], self.weights[3])
        pool2 = ch.max_pool(conv2)

        # Fully Connected 1
        fc1 = ch.fully_connected_as_corr(pool2, self.weights[4], self.weights[5])

        # Output layer
        prob = np.dot(fc1, self.weights[6])
        prob += self.weights[7][np.newaxis, :]
        prob = prob - np.amax(prob, axis=1, keepdims=True)
        prob = np.exp(prob)
        prob = prob[:, 1] / prob.sum(1)
        prob = np.hstack((prob, np.zeros((ip.shape[1]//4)-prob.shape[0], dtype=np.float32)))

        return prob


    def eval_network_2_dense(self, ip):
        """cnn with 2 dense layers at end
        """

        # Conv Layer 1
        conv1 = ch.corr2d(ip[np.newaxis,:], self.weights[0], self.weights[1])
        pool1 = ch.max_pool(conv1)

        # Conv Layer 2
        conv2 = ch.corr2d(pool1, self.weights[2], self.weights[3])
        pool2 = ch.max_pool(conv2)

        # Fully Connected 1
        fc1 = ch.fully_connected_as_corr(pool2, self.weights[4], self.weights[5])

        # Fully Connected 2
        fc2 = np.dot(fc1, self.weights[6])  # fc times fc
        fc2 += self.weights[7][np.newaxis, :]  # add bias term
        fc2.clip(min=0, out=fc2)  # non linearity - ReLu

        # Output layer
        prob = np.dot(fc2, self.weights[8])
        prob += self.weights[9][np.newaxis, :]
        prob = prob - np.amax(prob, axis=1, keepdims=True)
        prob = np.exp(prob)
        prob = prob[:, 1] / prob.sum(1)
        prob = np.hstack((prob, np.zeros((ip.shape[1]//4)-prob.shape[0], dtype=np.float32)))

        return prob

