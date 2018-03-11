"""
This script evaluates the performance of the CPU version of CNN_FAST on the
different test sets.

First you need to run 'run_detector.py' with these settings to produce
'results/op_file.csv':

detection_thresh = 0.0
do_time_expansion = False
root_dir = '../bat_train/data/'  # set this to where the annotations and wav files are
data_set = root_dir + 'train_test_split/test_set_bulgaria.npz'
data_dir = root_dir + 'wav/'
loaded_data_tr = np.load(data_set)
audio_files = loaded_data_tr['test_files']
audio_files = [data_dir + tt + '.wav' for tt in audio_files]
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('../bat_train/')
import evaluate as evl
import create_results as res

# point to data
root_dir = '../bat_train/data/'

# load the test data
data_set = root_dir + 'train_test_split/test_set_uk.npz'
loaded_data_tr = np.load(data_set)
test_pos = loaded_data_tr['test_pos']
test_files = loaded_data_tr['test_files']
test_durations = loaded_data_tr['test_durations']


# load results and put them in the correct format
da = pd.read_csv('results/op_file.csv')
nms_pos = []
nms_prob = []
for tt in test_files:
    dal = da[da['file_name'] == tt+'.wav']
    nms_pos.append(dal['detection_time'].values)
    nms_prob.append(dal['detection_prob'].values[..., np.newaxis])


# compute precision and recall
precision, recall = evl.prec_recall_1d(nms_pos, nms_prob, test_pos, test_durations, 0.1, 0.230)
res.plot_prec_recall('CNN_FAST', recall, precision, nms_prob)

