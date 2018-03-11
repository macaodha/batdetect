import time
import numpy as np


class DataSetParams:

    def __init__(self):

        # spectrogram generation
        self.spectrogram_params()

        # detection
        self.detection()

        # data
        self.spec_dir = ''
        self.audio_dir = ''

        self.save_features_to_disk = False
        self.load_features_from_file = False

        # hard negative mining
        self.num_hard_negative_mining = 2  # if 0 there won't be any

        # non max suppression - smoothing and window
        self.smooth_op_prediction = True  # smooth the op parameters before nms
        self.smooth_op_prediction_sigma = 0.006 / self.time_per_slice
        self.nms_win_size = int(np.round(0.12 / self.time_per_slice))  #ie 21 samples at 0.02322 fft win size, 0.75 overlap

        # model
        self.classification_model = 'cnn'  # rf_vanilla, segment, cnn

        # rf_vanilla params
        self.feature_type = 'grad_pool'  # raw, grad, grad_pool, raw_pool, hog, max_freq
        self.trees = 50
        self.depth = 20
        self.min_cnt = 2
        self.tests = 5000

        # CNN params
        self.learn_rate = 0.01
        self.moment = 0.9
        self.num_epochs = 50
        self.batchsize = 256
        self.net_type = 'big'  # big, small

        # segment params - these were cross validated on validation set
        self.median_mult = 5.0  # how much to treshold spectrograms - higher will mean less calls
        self.min_region_size = np.round(0.4/self.time_per_slice)  # used to determine the thresholding - 65 for fft win 0.02322
        self.min_overlap = 0.1  # in secs, anything that overlaps by this much will be counted as 1 call

        # param name string
        self.model_identifier = time.strftime("%d_%m_%y_%H_%M_%S_") + self.classification_model + '_hnm_' + str(self.num_hard_negative_mining)
        if self.classification_model == 'rf_vanilla':
            self.model_identifier += '_feat_' + self.feature_type
        elif self.classification_model == 'cnn':
            self.model_identifier += '_lr_'+ str(self.learn_rate) + '_mo_'+ str(self.moment) + '_net_'+ self.net_type
        elif self.classification_model == 'segment':
            self.model_identifier += '_minSize_' + str(self.min_region_size) + '_minOverlap_' + str(self.min_overlap )

        # misc
        self.run_parallel = True
        self.num_processes = 10
        self.add_extra_calls = True  # sample some other positive calls near the GT
        self.aug_shift = 0.015  # unit seconds, add extra call either side of GT if augmenting

    def spectrogram_params(self):

        self.valid_file_length = 169345  # some files are longer than they should be

        # spectrogram generation
        self.fft_win_length = 0.02322  # ie 1024/44100.0 about 23 msecs.
        self.fft_overlap = 0.75  # this is a percent - previously was 768/1024
        self.time_per_slice = ((1-self.fft_overlap)*self.fft_win_length)

        self.denoise = True
        self.mean_log_mag = 0.5  # sensitive to the spectrogram scaling used
        self.smooth_spec = True  # gaussian filter

        # throw away unnecessary frequencies, keep from bottom
        # TODO this only makes sense as a frequency when you know the sampling rate
        # better to think of these as indices
        self.crop_spec = True
        self.max_freq = 270
        self.min_freq = 10

        # if doing 192K files for training
        #self.fft_win_length = 0.02667  # i.e. 512/19200
        #self.max_freq = 240
        #self.min_freq = 10

    def detection(self):
        self.window_size = 0.230  # 230 milliseconds (in time expanded, so 23 ms for not)
        # represent window size in terms of the number of time bins
        self.window_width = np.rint(self.window_size / ((1-self.fft_overlap)*self.fft_win_length))
        self.detection_overlap = 0.1  # needs to be within x seconds of GT to be considered correct
        self.detection_prob = 0.5  # everything under this is considered background - used in HNM
