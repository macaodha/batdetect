import evaluate as evl
import matplotlib.pyplot as plt
import numpy as np
import os
import spectrogram as sp
from scipy.io import wavfile
import seaborn as sns
sns.set_style('whitegrid')


def plot_prec_recall(alg_name, recall, precision, nms_prob=None):
    # average precision
    ave_prec = evl.calc_average_precision(recall, precision)
    print 'average precision (area) = %.3f ' % ave_prec

    # recall at 95% precision
    desired_precision = 0.95
    if np.where(precision >= desired_precision)[0].shape[0] > 0:
        recall_at_precision = recall[np.where(precision >= desired_precision)[0][-1]]
    else:
        recall_at_precision = 0

    print 'recall at', int(desired_precision*100), '% precision = ', "%.3f" % recall_at_precision
    plt.plot([0, 1.02], [desired_precision, desired_precision], 'b:', linewidth=1)
    plt.plot([recall_at_precision, recall_at_precision], [0, desired_precision], 'b:', linewidth=1)

    # create plot
    label_str = alg_name.ljust(8) + "%.3f" % ave_prec + '  ' + str(desired_precision) + ' rec %.3f' % recall_at_precision
    if recall.shape[0] == 1:
        plt.plot(recall, precision, 'o', label=label_str)
    else:
        plt.plot(recall, precision, '', label=label_str)

    # find different probability locations on curve
    if nms_prob is not None:
        conf = np.concatenate(nms_prob)[:, 0]
        for p_val in [0.9, 0.7, 0.5]:
            p_loc = np.where(np.sort(conf)[::-1] < p_val)[0]
            if p_loc.shape[0] > 0:
                plt.plot(recall[p_loc[0]], precision[p_loc[0]], 'o', color='#4C72B0')
                plt.text(recall[p_loc[0]]-0.05, precision[p_loc[0]]-0.05, str(p_val))

    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.axis((0, 1.02, 0, 1.02))
    plt.legend(loc='lower left')
    plt.grid(1)
    plt.show()


def plot_spec(op_file_name, ip_file, gt_pos, nms_pos, nms_prob, y_prediction, params, save_ims):

    # create spec
    sampling_rate, audio_samples = wavfile.read(ip_file)
    file_duration = audio_samples.shape[0] / float(sampling_rate)
    spectrogram = sp.gen_spectrogram(audio_samples, sampling_rate, params.fft_win_length, params.fft_overlap,
                                     crop_spec=params.crop_spec, max_freq=params.max_freq, min_freq=params.min_freq)

    if y_prediction is None:
        y_prediction = np.zeros((spectrogram.shape[1]))

    gt_pos_norm = (gt_pos/file_duration)*y_prediction.shape[0]
    nms_pos_norm = (nms_pos/file_duration)*y_prediction.shape[0]

    fig = plt.figure(1, figsize=(10, 6))
    ax1 = plt.axes([0.05, 0.7, 0.9, 0.25])
    ax0 = plt.axes([0.05, 0.05, 0.9, 0.60])

    ax1.plot([0, y_prediction.shape[0]], [0.5, 0.5], 'k--', linewidth=0.5, label='pred')

    # plot gt
    for pt in gt_pos_norm:
        ax1.plot([pt, pt], [0, 1], 'g', linewidth=4, label='gt')

    # plot nms
    for p in range(len(nms_pos_norm)):
        ax1.plot([nms_pos_norm[p], nms_pos_norm[p]], [0, nms_prob[p]], 'r', linewidth=2, label='pred')

    ax1.plot(y_prediction)
    ax1.set_xlim(0, y_prediction.shape[0])
    ax1.set_ylim(0, 1)
    ax1.xaxis.set_ticklabels([])

    # plot image
    ax0.imshow(spectrogram, aspect='auto', cmap='plasma')
    ax0.xaxis.set_ticklabels([])
    ax0.yaxis.set_ticklabels([])
    plt.grid()

    if save_ims:
        fig.savefig(op_file_name + '.jpg')

    plt.close(1)
