from __future__ import print_function
import numpy as np
import os
import fnmatch
import time
import sys

import write_op as wo
import cpu_detection as detector
import mywavfile


def get_audio_files(ip_dir):
    matches = []
    for root, dirnames, filenames in os.walk(ip_dir):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                matches.append(os.path.join(root, filename))
    return matches


def read_audio(file_name, do_time_expansion, chunk_size, win_size):
    # try to read in audio file
    try:
        samp_rate_orig, audio = mywavfile.read(file_name)
    except:
        print('  Error reading file')
        return True, None, None, None, None

    # convert to mono if stereo
    if len(audio.shape) == 2:
        print('  Warning: stereo file. Just taking left channel.')
        audio = audio[:, 0]
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


def run_model(det, audio, file_dur, samp_rate, detection_thresh, max_num_calls=0):
    """This runs the bat call detector.
    """
    # results will be stored here
    det_time_file = np.zeros(0)
    det_prob_file = np.zeros(0)

    # files can be long so we split each up into separate (overlapping) chunks
    st_positions = np.arange(0, file_dur, det.chunk_size-det.win_size)
    for chunk_id, st_position in enumerate(st_positions):

        # take a chunk of the audio
        # should already be zero padded at the end so its the correct size
        st_pos = int(st_position*samp_rate)
        en_pos = int(st_pos + det.chunk_size*samp_rate)
        audio_chunk = audio[st_pos:en_pos]
        chunk_duration = audio_chunk.shape[0] / float(samp_rate)

        # create spectrogram
        spec = det.create_spec(audio_chunk, samp_rate)

        # run detector
        det_loc, prob_det = det.run_detection(spec, chunk_duration, detection_thresh,
                                              low_res=True)

        det_time_file = np.hstack((det_time_file, det_loc + st_position))
        det_prob_file = np.hstack((det_prob_file, prob_det))

    # undo the effects of time expansion for detector
    if do_time_expansion:
        det_time_file /= 10.0

    return det_time_file, det_prob_file


if __name__ == "__main__":

    # params
    detection_thresh        = 0.95 # make this smaller if you want more calls
    do_time_expansion       = True # if audio is already time expanded set this to False
    save_individual_results = True # if True will create an output for each file
    save_summary_result     = True # if True will create a single csv file with all results

    # load data
    data_dir   = 'wavs'            # this is the path to your audio files
    op_ann_dir = 'results'         # this where your results will be saved
    op_ann_dir_ind     = os.path.join(op_ann_dir, 'individual_results')  # this where individual results will be saved
    op_file_name_total = os.path.join(op_ann_dir, 'results.csv')
    if not os.path.isdir(op_ann_dir):
        os.makedirs(op_ann_dir)
    if save_individual_results and not os.path.isdir(op_ann_dir_ind):
        os.makedirs(op_ann_dir_ind)

    # read audio files
    audio_files = get_audio_files(data_dir)

    print('Processing        ', len(audio_files), 'files')
    print('Input directory   ', data_dir)
    print('Results directory ', op_ann_dir, '\n')


    # load and create the detector
    det_model_file  = 'models/detector.npy'
    det_params_file = det_model_file[:-4] + '_params.json'
    det = detector.CPUDetector(det_model_file, det_params_file)

    # loop through audio files
    results = []
    for file_cnt, file_name in enumerate(audio_files):

        file_name_basename = file_name[len(data_dir):]
        print('\n', file_cnt+1, 'of', len(audio_files), '\t', file_name_basename)

        # read audio file - skip file if can't read it
        read_fail, audio, file_dur, samp_rate, samp_rate_orig = read_audio(file_name,
                                do_time_expansion, det.chunk_size, det.win_size)
        if read_fail:
            continue

        # run detector
        tic = time.time()
        det_time, det_prob  = run_model(det, audio, file_dur, samp_rate,
                                        detection_thresh)
        toc = time.time()

        print('  detection time', round(toc-tic, 3), '(secs)')
        num_calls = len(det_time)
        print('  ' + str(num_calls) + ' calls found')

        # save results
        if save_individual_results:
            # save to AudioTagger format
            f_name_fmt   = file_name_basename.replace('/', '_').replace('\\', '_')[:-4]
            op_file_name = os.path.join(op_ann_dir_ind, f_name_fmt) + '-sceneRect.csv'
            wo.create_audio_tagger_op(file_name_basename, op_file_name, det_time,
                                      det_prob, samp_rate_orig, class_name='bat')

        # save as dictionary
        if num_calls > 0:
            res = {'filename':file_name_basename, 'time':det_time, 'prob':det_prob}
            results.append(res)

    # save results for all files to large csv
    if save_summary_result and (len(results) > 0):
        print('\nsaving results to', op_file_name_total)
        wo.save_to_txt(op_file_name_total, results)
    else:
        print('no detections to save')
