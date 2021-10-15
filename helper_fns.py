import numpy as np
from skimage import filters
from skimage.util.shape import view_as_windows
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter1d
from scipy.io import wavfile

fft_win_length    = 0.02322
fft_overlap       = 0.75
crop_spec         = True
max_freq          = 270
min_freq          = 10
denoise           = True
mean_log_mag      = 0.5
smooth_spec       = True
window_size       = 0.230  
window_width      = np.rint(window_size / ((1-fft_overlap)*fft_win_length))
detection_overlap = 0.1 
detection_prob    = 0.5

save_features_to_disk   = True
load_features_from_file = False

spec_dir  = ''
audio_dir = ''

def generate_training_positions(files, gt_pos, durations):
    positions    = [None]*len(files)
    class_labels = [None]*len(files)
    for ii, ff in enumerate(files):
        positions[ii], class_labels[ii] = extract_train_position_from_file(gt_pos[ii], durations[ii])
    return positions, class_labels

def extract_train_position_from_file(gt_pos, duration):
    """
    Samples random negative locations for negs, making sure not to overlap with GT.

    gt_pos is the time in seconds that the call occurs.
    positions contains time in seconds of some negative and positive examples.
    """

    if gt_pos.shape[0] == 0:
        # dont extract any values if the file does not contain anything
        # we will use these ones for HNM later
        positions    = np.zeros(0)
        class_labels = np.zeros((0,1))
    else:
        shift         = 0  # if there is augmentation this is how much we will add
        num_neg_calls = gt_pos.shape[0]
        window_size   = 0.230
        pos_window    = window_size / 2 # window around GT that is not sampled from
        pos           = gt_pos[:, 0]

        # augmentation
        add_extra_calls = True
        if add_extra_calls:
            shift = 0.015
            num_neg_calls *= 3
            pos = np.hstack((gt_pos[:, 0] - shift, gt_pos[:, 0], gt_pos[:, 0] + shift))

        # sample a set of negative locations - need to be sufficiently far away from GT
        pos_pad = np.hstack((0-window_size, gt_pos[:, 0], duration-window_size))
        neg     = []
        cnt     = 0
        while cnt < num_neg_calls:
            rand_pos = np.random.random()*pos_pad.max()
            if (np.abs(pos_pad - rand_pos) > (pos_window+shift)).mean() == 1:
                neg.append(rand_pos)
                cnt += 1
        neg = np.asarray(neg)

        # sort them
        positions   = np.hstack((pos, neg))
        sorted_inds = np.argsort(positions)
        positions   = positions[sorted_inds]

        # create labels
        class_labels = np.vstack((np.ones((pos.shape[0], 1)), np.zeros((neg.shape[0], 1))))
        class_labels = class_labels[sorted_inds]

    return positions, class_labels

def denoise_fn(spec_noisy, mask=None):
    """
    Perform denoising, subtract mean from each frequency band.
    Mask chooses the relevant time steps to use.
    """
    
    if mask is None:
        # no mask
        me = np.mean(spec_noisy, 1)
        spec_denoise = spec_noisy - me[:, np.newaxis]

    else:
        # user defined mask
        mask_inv     = np.invert(mask)
        spec_denoise = spec_noisy.copy()

        if np.sum(mask) > 0:
            me = np.mean(spec_denoise[:, mask], 1)
            spec_denoise[:, mask] = spec_denoise[:, mask] - me[:, np.newaxis]

        if np.sum(mask_inv) > 0:
            me_inv = np.mean(spec_denoise[:, mask_inv], 1)
            spec_denoise[:, mask_inv] = spec_denoise[:, mask_inv] - me_inv[:, np.newaxis]

    # remove anything below 0
    spec_denoise.clip(min=0, out=spec_denoise)

    return spec_denoise

def gen_mag_spectrogram_fft(x, nfft, noverlap):
    """
    Compute magnitude spectrogram by specifying num bins.
    """

    # window data
    step    = nfft - noverlap
    shape   = (nfft, (x.shape[-1]-noverlap)//step)
    strides = (x.strides[0], step*x.strides[0])
    x_wins  = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # apply window
    x_wins_han = np.hanning(x_wins.shape[0])[..., np.newaxis] * x_wins

    # do fft
    complex_spec = np.fft.rfft(x_wins_han, n=nfft, axis=0)

    # calculate magnitude
    mag_spec = np.conjugate(complex_spec) * complex_spec
    mag_spec = mag_spec.real
    # same as:
    #mag_spec = np.square(np.absolute(complex_spec))

    # orient correctly and remove dc component
    mag_spec = mag_spec[1:, :]
    mag_spec = np.flipud(mag_spec)

    return mag_spec

def gen_mag_spectrogram(x, fs, ms, overlap_perc):
    """
    Computes magnitude spectrogram by specifying time.
    """
    
    nfft     = int(ms*fs)
    noverlap = int(overlap_perc*nfft)

    # window data
    step    = nfft - noverlap
    shape   = (nfft, (x.shape[-1]-noverlap)//step)
    strides = (x.strides[0], step*x.strides[0])
    x_wins  = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # apply window
    x_wins_han = np.hanning(x_wins.shape[0])[..., np.newaxis] * x_wins

    # do fft
    # note this will be much slower if x_wins_han.shape[0] is not a power of 2
    complex_spec = np.fft.rfft(x_wins_han, axis=0)

    # calculate magnitude
    mag_spec = (np.conjugate(complex_spec) * complex_spec).real
    # same as:
    #mag_spec = np.square(np.absolute(complex_spec))

    # orient correctly and remove dc component
    spec = mag_spec[1:, :]
    spec = np.flipud(spec)

    return spec

def gen_spectrogram(audio_samples, sampling_rate, fft_win_length, fft_overlap, crop_spec=True, max_freq=256, min_freq=0):
    """
    Compute spectrogram, crop and compute log.
    """

    # compute spectrogram
    spec = gen_mag_spectrogram(audio_samples, sampling_rate, fft_win_length, fft_overlap)

    # only keep the relevant bands - could do this outside
    if crop_spec:
        spec = spec[-max_freq:-min_freq, :]

        # add some zeros if too small
        req_height = max_freq-min_freq
        if spec.shape[0] < req_height:
            zero_pad = np.zeros((req_height-spec.shape[0], spec.shape[1]))
            spec     = np.vstack((zero_pad, spec))

    # perform log scaling - here the same as matplotlib
    log_scaling = 2.0 * (1.0 / sampling_rate) * (1.0/(np.abs(np.hanning(int(fft_win_length*sampling_rate)))**2).sum())
    spec        = np.log(1.0 + log_scaling*spec)

    return spec

def process_spectrogram(spec, denoise_spec=True, mean_log_mag=0.5, smooth_spec=True):
    """
    Denoises, and smooths spectrogram.
    """
    
    # denoise
    if denoise_spec:
        # use a mask as there is silence at the start and end of recs
        mask = spec.mean(0) > mean_log_mag
        spec = denoise_fn(spec, mask)

    # smooth the spectrogram
    if smooth_spec:
        spec = filters.gaussian(spec, 1.0)

    return spec

def create_or_load_features(file_name=None, audio_samples=None, sampling_rate=None):
    """
    Does 1 of 3 possible things
    1) computes feature from audio samples directly
    2) loads feature from disk OR
    3) computes features from file name
    """

    if file_name is None:
        features = compute_features(audio_samples, sampling_rate)
    else:
        if load_features_from_file:
            features = np.load(audio_dir + file_name + '.npy')
        else:
            sampling_rate, audio_samples = wavfile.read(audio_dir + file_name + '.wav')
            features = compute_features(audio_samples, sampling_rate)

    return features

def compute_features(audio_samples, sampling_rate):
    """
    Computes overlapping windows of spectrogram as input for CNN.
    """

    # load audio and create spectrogram
    spectrogram = gen_spectrogram(audio_samples, sampling_rate, fft_win_length, fft_overlap,
                                     crop_spec=crop_spec, max_freq=max_freq, min_freq=min_freq)
    spectrogram = process_spectrogram(spectrogram, denoise_spec=denoise, mean_log_mag=mean_log_mag, smooth_spec=smooth_spec)

    # extract windows
    spec_win   = view_as_windows(spectrogram, (spectrogram.shape[0], window_width))[0]
    spec_win   = zoom(spec_win, (1, 0.5, 0.5), order=1)
    spec_width = spectrogram.shape[1]

    # make the correct size for CNN
    features = np.zeros((spec_width, 1, spec_win.shape[1], spec_win.shape[2]), dtype=np.float32)
    features[:spec_win.shape[0], 0, :, :] = spec_win

    return features


def get_audio_features_and_labels(class_labels, positions, durations, paths_decode):
    feats = []
    labs  = []
    count = 1
    for ilab, ipos, idur, file_name in zip(class_labels, positions, durations, paths_decode):
        if count % 50 == 0:
            print(count, '/', len(paths_decode))
        count = count + 1
        if ipos.shape[0] > 0:
            local_feats = create_or_load_features(file_name)

            # convert time in file to integer
            positions_ratio = ipos / idur
            inds = (positions_ratio*float(local_feats.shape[0])).astype('int')

            feats.append(local_feats[inds, :, :, :])
            labs.append(ilab)
    features = np.vstack(feats)
    features = np.squeeze(features)
    labels   = np.vstack(labs).astype(np.uint8)[:,0]
    return features, labels

