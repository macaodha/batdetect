from skimage import filters
import numpy as np


def denoise(spec_noisy, mask=None):
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
        mask_inv = np.invert(mask)
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
    step = nfft - noverlap
    shape = (nfft, (x.shape[-1]-noverlap)//step)
    strides = (x.strides[0], step*x.strides[0])
    x_wins = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

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

    nfft = int(ms*fs)
    noverlap = int(overlap_perc*nfft)

    # window data
    step = nfft - noverlap
    shape = (nfft, (x.shape[-1]-noverlap)//step)
    strides = (x.strides[0], step*x.strides[0])
    x_wins = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

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
            spec = np.vstack((zero_pad, spec))

    # perform log scaling - here the same as matplotlib
    log_scaling = 2.0 * (1.0 / sampling_rate) * (1.0/(np.abs(np.hanning(int(fft_win_length*sampling_rate)))**2).sum())
    spec = np.log(1.0 + log_scaling*spec)

    return spec


def process_spectrogram(spec, denoise_spec=True, mean_log_mag=0.5, smooth_spec=True):
    """
    Denoises, and smooths spectrogram.
    """

    # denoise
    if denoise_spec:
        # use a mask as there is silence at the start and end of recs
        mask = spec.mean(0) > mean_log_mag
        spec = denoise(spec, mask)

    # smooth the spectrogram
    if smooth_spec:
        spec = filters.gaussian(spec, 1.0)

    return spec
