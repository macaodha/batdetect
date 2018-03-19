"""
This script outputs the weights of a trained model so the standalone detector copde
can use it.
"""

import cPickle as pickle
from lasagne.layers.helper import get_all_param_values, get_output_shape, set_all_param_values
import numpy as np
import lasagne
import theano
import sys
import cPickle as pickle
import json

save_detector = False

print 'saving detector'
model_dir = 'results/models/'
model_file = model_dir + 'test_set_norfolk.mod'
print model_file

mod = pickle.load(open(model_file))

weights = get_all_param_values(mod.model.network['prob'])
np.save(model_file[:-4], weights)
print 'weights shape', len(weights)

# save detection params
mod_params = {'win_size':0, 'chunk_size':0, 'max_freq':0, 'min_freq':0,
              'mean_log_mag':0, 'slice_scale':0, 'overlap':0,
              'crop_spec':False, 'denoise':False, 'smooth_spec':False,
              'nms_win_size':0, 'smooth_op_prediction_sigma':0}

mod_params['win_size'] = mod.model.params.window_size
mod_params['max_freq'] = mod.model.params.max_freq
mod_params['min_freq'] = mod.model.params.min_freq
mod_params['mean_log_mag'] = mod.model.params.mean_log_mag
mod_params['slice_scale'] = mod.model.params.fft_win_length
mod_params['overlap'] = mod.model.params.fft_overlap

mod_params['crop_spec'] = mod.model.params.crop_spec
mod_params['denoise'] = mod.model.params.denoise
mod_params['smooth_spec'] = mod.model.params.smooth_spec

mod_params['nms_win_size'] = int(mod.model.params.nms_win_size)
mod_params['smooth_op_prediction_sigma'] = mod.model.params.smooth_op_prediction_sigma

params_file = model_file[:-4] + '_params.p'
with open(params_file, 'w') as fp:
    json.dump(mod_params, fp)
