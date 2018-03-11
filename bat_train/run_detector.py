from scipy.io import wavfile
import numpy as np
import cPickle as pickle
import os
import glob
import time
import write_op as wo
import sys


def read_audio(file_name, do_time_expansion, chunk_size, win_size):

    # try to read in audio file
    try:
        samp_rate_orig, audio = wavfile.read(file_name)
    except:
        print '  Error reading file'
        return True, None, None, None, None

    # convert to mono if stereo
    if len(audio.shape) == 2:
        print '  Warning: stereo file. Just taking right channel.'
        audio = audio[:, 1]
    file_dur = audio.shape[0] / float(samp_rate_orig)
    print '  dur', round(file_dur,3), '(secs) , fs', samp_rate_orig

    # original model is trained on time expanded data
    samp_rate = samp_rate_orig
    if do_time_expansion:
        samp_rate = int(samp_rate_orig/10.0)
        file_dur *= 10

    # pad with zeros so we can go right to the end
    multiplier = np.ceil(file_dur/float(chunk_size-win_size))
    diff = multiplier*(chunk_size-win_size) - file_dur + win_size
    audio_pad = np.hstack((audio, np.zeros(int(diff*samp_rate))))

    return False, audio_pad, file_dur, samp_rate, samp_rate_orig


def run_detector(det, audio, file_dur, samp_rate, detection_thresh):

    det_time = []
    det_prob = []

    # files can be long so we split each up into separate (overlapping) chunks
    st_positions = np.arange(0, file_dur, det.chunk_size-det.params.window_size)
    for chunk_id, st_position in enumerate(st_positions):

        # take a chunk of the audio
        # should already be zero padded at the end so its the correct size
        st_pos = int(st_position*samp_rate)
        en_pos = int(st_pos + det.chunk_size*samp_rate)
        audio_chunk = audio[st_pos:en_pos]

        # make predictions
        pos, prob, y_prediction = det.test_single(audio_chunk, samp_rate)
        prob = prob[:, 0]

        # remove predictions near the end (if not last chunk) and ones that are
        # below the detection threshold
        if chunk_id == (len(st_positions)-1):
            inds = (prob >= detection_thresh)
        else:
            inds = (prob >= detection_thresh) & (pos < (det.chunk_size-(det.params.window_size/2.0)))

        # convert detection time back into global time and save valid detections
        if pos.shape[0] > 0:
            det_time.append(pos[inds] + st_position)
            det_prob.append(prob[inds])

    if len(det_time) > 0:
        det_time = np.hstack(det_time)
        det_prob = np.hstack(det_prob)

        # undo the effects of times expansion
        if do_time_expansion:
            det_time /= 10.0

    return det_time, det_prob


if __name__ == "__main__":
    """
    This code takes a directory of audio files and runs a CNN based bat call
    detector. It returns the time in file of the detection and the probability
    that the detection is a bat call.
    """

    # params
    detection_thresh = 0.80   # make this smaller if you want more calls detected
    do_time_expansion = True  # set to True if audio is not already time expanded
    save_res = True

    # load data -
    data_dir = 'path_to_data/' # path of the data that we run the model on
    op_ann_dir = 'results/'    # where we will store the outputs
    op_file_name_total = op_ann_dir + 'op_file.csv'
    if not os.path.isdir(op_ann_dir):
        os.makedirs(op_ann_dir)

    # load gpu lasagne model
    model_dir = 'data/models/'
    model_file = model_dir + 'test_set_bulgaria.mod'
    det = pickle.load(open(model_file))
    det.chunk_size = 4.0

    # read audio files
    audio_files = glob.glob(data_dir + '*.wav')

    # loop through audio files
    results = []
    for file_cnt, file_name in enumerate(audio_files):

        file_name_root = file_name[len(data_dir):]
        print '\n', file_cnt+1, 'of', len(audio_files), '\t', file_name_root

        # read audio file - skip file if cannot read
        read_fail, audio, file_dur, samp_rate, samp_rate_orig = read_audio(file_name,
                                do_time_expansion, det.chunk_size, det.params.window_size)
        if read_fail:
            continue

        # run detector
        tic = time.time()
        det_time, det_prob = run_detector(det, audio, file_dur, samp_rate,
                                          detection_thresh)
        toc = time.time()

        print '  detection time', round(toc-tic, 3), '(secs)'
        num_calls = len(det_time)
        print '  ' + str(num_calls) + ' calls found'

        # save results
        if save_res:
            # return detector results
            pred_classes = np.zeros((len(det_time), 1), dtype=np.int)
            pred_prob = np.asarray(det_prob)[..., np.newaxis]

            # save to AudioTagger format
            op_file_name = op_ann_dir + file_name_root[:-4] + '-sceneRect.csv'
            wo.create_audio_tagger_op(file_name_root, op_file_name, det_time,
                                      det_prob, pred_classes[:,0], pred_prob[:,0],
                                      samp_rate_orig, np.asarray(['bat']))

            # save as dictionary
            if num_calls > 0:
                res = {'filename':file_name_root, 'time':det_time,
                       'prob':det_prob, 'pred_classes':pred_classes,
                       'pred_prob':pred_prob}
                results.append(res)

    # save to large csv
    if save_res and (len(results) > 0):
        print '\nsaving results to', op_file_name_total
        wo.save_to_txt(op_file_name_total, results, np.asarray(['bat']))
    else:
        print 'no detections to save'
