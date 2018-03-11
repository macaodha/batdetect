from myskimage import gaussian
import numpy as np
import imp
try:
    imp.find_module('pyfftw')
    pyfftw_installed = True
    import pyfftw
except ImportError:
    pyfftw_installed = False


class Spectrogram:
    fftw_inps = {}
    fftw_rfft = {}
    han_wins = {}

    def __init__(self, use_pyfftw=True):
        if not pyfftw_installed:
            use_pyfftw = False
        self.use_pyfftw = use_pyfftw

    @staticmethod
    def _denoise(spec):
        """
        Perform denoising.
        """
        me = np.mean(spec, 1)
        spec = spec - me[:, np.newaxis]

        # remove anything below 0
        spec.clip(min=0, out=spec)

        return spec

    @staticmethod
    def do_fft(inp, use_pyfftw=False, K=None):
        if not use_pyfftw:
            out = np.fft.rfft(inp, n=K, axis=0)
            out = out.astype('complex64') # numpy may be using double precision internally
        elif use_pyfftw:
            if not inp.shape in Spectrogram.fftw_inps:
                Spectrogram.fftw_inps[inp.shape] = pyfftw.empty_aligned(inp.shape, dtype='float32')
                Spectrogram.fftw_rfft[inp.shape] = pyfftw.builders.rfft(Spectrogram.fftw_inps[inp.shape], axis=0)
            Spectrogram.fftw_inps[inp.shape][:] = inp[:]
            out = (Spectrogram.fftw_rfft[inp.shape])()
        return out

    def gen_mag_spectrogram(self, x, fs, ms, overlap_perc, crop_spec=True, max_freq=256, min_freq=0):
        """
        Computes magnitude spectrogram by specifying time
        """

        x = x.astype(np.float32)

        nfft = int(ms*fs)
        noverlap = int(overlap_perc*nfft)

        # window data
        step = nfft - noverlap
        shape = (nfft, (x.shape[-1]-noverlap)//step)
        strides = (x.strides[0], step*x.strides[0])
        x_wins = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        # apply window
        if x_wins.shape not in Spectrogram.han_wins:
            Spectrogram.han_wins[x_wins.shape[0]] = np.hanning(x_wins.shape[0]).astype('float32')

        han_win = Spectrogram.han_wins[x_wins.shape[0]]
        x_wins_han = han_win[..., np.newaxis] * x_wins

        # do fft
        # note this will be much slower if x_wins_han.shape[0] is not a power of 2
        complex_spec = Spectrogram.do_fft(x_wins_han, self.use_pyfftw)

        # calculate magnitude
        mag_spec = complex_spec.real**2 + complex_spec.imag**2
        # calculate magnitude
        #mag_spec = (np.conjugate(complex_spec) * complex_spec).real
        # calculate magnitude
        #mag_spec = np.square(np.absolute(complex_spec))

        # orient correctly and remove dc component
        spec = mag_spec[1:, :]
        spec = np.flipud(spec)

        # only keep the relevant bands
        # not really in frequency, better thought of as indices
        if crop_spec:
            spec = spec[-max_freq:-min_freq, :]

            # add some zeros if too small
            req_height = max_freq-min_freq
            if spec.shape[0] < req_height:
                zero_pad = np.zeros((req_height-spec.shape[0], spec.shape[1]), dtype=np.float32)
                spec = np.vstack((zero_pad, spec))
        return spec


    def gen_spectrogram(self, audio_samples, sampling_rate, fft_win_length, fft_overlap, crop_spec=True, max_freq=256, min_freq=0):
        """
        Compute spectrogram, crop and compute log.
        """

        # compute spectrogram
        spec = self.gen_mag_spectrogram(audio_samples, sampling_rate, fft_win_length, fft_overlap, crop_spec, max_freq, min_freq)

        # perform log scaling - here the same as matplotlib
        log_scaling = 2.0 * (1.0 / sampling_rate) * (1.0/(np.abs(np.hanning(int(fft_win_length*sampling_rate)))**2).sum())
        spec = np.log(1 + log_scaling*spec)

        return spec


    def process_spectrogram(self, spec, denoise_spec=True, smooth_spec=True, smooth_sigma=1.0):
        """
        Denoises, and smooths spectrogram.
        """

        # denoise
        if denoise_spec:
            spec = Spectrogram._denoise(spec)

        # smooth the spectrogram
        if smooth_spec:
            spec = gaussian(spec, smooth_sigma)

        return spec
