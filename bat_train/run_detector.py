from scipy.io import wavfile
import numpy as np
import pickle
import os
import glob
import time
import write_op as wo
from data_set_params import DataSetParams
import sys
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from helper_fns import compute_features, nms_1d
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


def read_audio(file_name, do_time_expansion, chunk_size, win_size):

    # try to read in audio file
    try:
        samp_rate_orig, audio = wavfile.read(file_name)
    except:
        print('  Error reading file')
        return True, None, None, None, None

    # convert to mono if stereo
    if len(audio.shape) == 2:
        print('  Warning: stereo file. Just taking right channel.')
        audio = audio[:, 1]
    file_dur = audio.shape[0] / float(samp_rate_orig)
    print('  dur', round(file_dur,3), '(secs) , fs', samp_rate_orig)

    # original model is trained on time expanded data
    samp_rate = samp_rate_orig
    if do_time_expansion:
        samp_rate = int(samp_rate_orig/10.0)
        file_dur *= 10

    # pad with zeros so we can go right to the end
    multiplier = np.ceil(file_dur/float(chunk_size-win_size))
    diff       = multiplier*(chunk_size-win_size) - file_dur + win_size
    audio_pad  = np.hstack((audio, np.zeros(int(diff*samp_rate))))

    return False, audio_pad, file_dur, samp_rate, samp_rate_orig


def run_detector(det, audio, file_dur, samp_rate, detection_thresh, params):

    det_time = []
    det_prob = []

    # files can be long so we split each up into separate (overlapping) chunks
    st_positions = np.arange(0, file_dur, params.chunk_size-params.window_size)
    #print('st_positions',st_positions)
    for chunk_id, st_position in enumerate(st_positions):

        # take a chunk of the audio
        # should already be zero padded at the end so its the correct size
        st_pos      = int(st_position*samp_rate)
        en_pos      = int(st_pos + params.chunk_size*samp_rate)
        audio_chunk = audio[st_pos:en_pos]
        # make predictions
        chunk_spec = compute_features(audio_chunk, samp_rate, params)
        chunk_spec = np.squeeze(chunk_spec)
        chunk_spec = np.expand_dims(chunk_spec,-1)
        
        det_pred = det.predict(chunk_spec)
        
        if params.smooth_op_prediction:
            det_pred = gaussian_filter1d(det_pred, params.smooth_op_prediction_sigma, axis=0)
        pos, prob = nms_1d(det_pred[:,0], params.nms_win_size, file_dur)
        #pos      = np.argmax(det_pred, axis=-1)
        prob = prob[:,0]
        #print('pos.shape', pos.shape)
        #print('prob.shape', prob.shape)
        #prob     = det_pred[:, 0]
        #print('(prob >= detection_thresh).shape', (prob >= detection_thresh).shape)
        #print('(pos < (params.chunk_size-(params.window_size/2.0)).shape',
        #(pos < (params.chunk_size-(params.window_size/2.0))).shape)
        #print((prob >= detection_thresh) & (pos < (params.chunk_size-(params.window_size/2.0))))
        #print(chunk_id)
        #print(len(st_positions)-1)
        # remove predictions near the end (if not last chunk) and ones that are
        # below the detection threshold
        if chunk_id == (len(st_positions)-1):
            inds = (prob >= detection_thresh)
        else:
            inds = np.logical_and((prob >= detection_thresh), (pos < (params.chunk_size-(params.window_size/2.0))))

        # convert detection time back into global time and save valid detections
        #print('inds.shape', inds.shape)
        #print('inds', inds)
        #print('st_position',st_position)
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
    detection_thresh  = 0.80  # make this smaller if you want more calls detected
    do_time_expansion = True  # set to True if audio is not already time expanded
    save_res = True

    params = DataSetParams()

    # load data -
    data_dir   = 'data/labelled_data/bulgaria/test/' # path of the data that we run the model on
    op_ann_dir = 'results/detections/'      # where we will store the outputs
    op_file_name_total = op_ann_dir + 'op_file.csv'
    if not os.path.isdir(op_ann_dir):
        os.makedirs(op_ann_dir)

    # load gpu lasagne model
    model_dir  = 'results/bulgaria_big_cnn'
    #model_file = model_dir + 'test_set_bulgaria.mod'
    #with open(model_file, 'rb') as mod_f:
    #    det = pickle.load(mod_f)
    det = tf.keras.models.load_model(model_dir)

    params.chunk_size = 4.0

    # read audio files
    audio_files = glob.glob(data_dir + '*.wav')[:100]
    #print(audio_files)
    # loop through audio files
    results = []
    for file_cnt, file_name in enumerate(audio_files):

        file_name_root = file_name[len(data_dir):]
        print('\n', file_cnt+1, 'of', len(audio_files), '\t', file_name_root)

        # read audio file - skip file if cannot read
        read_fail, audio, file_dur, samp_rate, samp_rate_orig = read_audio(file_name,
                                do_time_expansion, params.chunk_size, params.window_size)
        if read_fail:
            continue

        # run detector
        tic = time.time()
        det_time, det_prob = run_detector(det, audio, file_dur, samp_rate, detection_thresh, params)
        toc = time.time()

        print('  detection time', round(toc-tic, 3), '(secs)')
        num_calls = len(det_time)
        print('  ' + str(num_calls) + ' calls found')

        # save results
        if save_res:
            # return detector results
            pred_classes = np.zeros((len(det_time), 1), dtype=np.int)
            pred_prob    = np.asarray(det_prob)[..., np.newaxis]

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
        print('\nsaving results to', op_file_name_total)
        wo.save_to_txt(op_file_name_total, results, np.asarray(['bat']))
    else:
        print('no detections to save')
