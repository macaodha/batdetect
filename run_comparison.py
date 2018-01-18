import numpy as np
import matplotlib.pyplot as plt
import os
import evaluate as evl
import create_results as res
from data_set_params import DataSetParams
import classifier as clss
import pandas as pd
import cPickle as pickle


def read_baseline_res(baseline_file_name, test_files):
    da = pd.read_csv(baseline_file_name)
    pos = []
    prob = []
    for ff in test_files:
        rr = da[da['Filename'] == ff]
        inds = np.argsort(rr.TimeInFile.values)
        pos.append(rr.TimeInFile.values[inds])
        prob.append(rr.Quality.values[inds][..., np.newaxis])
    return pos, prob


if __name__ == '__main__':
    """
    This compares several different algorithms for bat echolocation detection.

    The results can vary by a few percent from run to run. If you don't want to
    run a specific model or baseline comment it out.
    """

    test_set = 'bulgaria'  # can be one of: bulgaria, uk, norfolk
    data_set = 'data/train_test_split/test_set_' + test_set + '.npz'
    raw_audio_dir = 'data/wav/'
    base_line_dir = 'data/baselines/'
    result_dir = 'results/'
    model_dir = 'data/models/'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    print 'test set:', test_set
    plt.close('all')

    # train and test_pos are in units of seconds
    loaded_data_tr = np.load(data_set)
    train_pos = loaded_data_tr['train_pos']
    train_files = loaded_data_tr['train_files']
    train_durations = loaded_data_tr['train_durations']
    test_pos = loaded_data_tr['test_pos']
    test_files = loaded_data_tr['test_files']
    test_durations = loaded_data_tr['test_durations']

    # load parameters
    params = DataSetParams()
    params.audio_dir = raw_audio_dir

    #
    # CNN
    print '\ncnn'
    params.classification_model = 'cnn'
    model = clss.Classifier(params)
    # train and test
    model.train(train_files, train_pos, train_durations)
    nms_pos, nms_prob = model.test_batch(test_files, test_pos, test_durations, False, '')
    # compute precision recall
    precision, recall = evl.prec_recall_1d(nms_pos, nms_prob, test_pos, test_durations, model.params.detection_overlap, model.params.window_size)
    res.plot_prec_recall('cnn', recall, precision, nms_prob)
    # save CNN model to file
    pickle.dump(model, open(model_dir + 'test_set_' + test_set + '.mod', 'wb'))

    #
    # random forest
    print '\nrandom forest'
    params.classification_model = 'rf_vanilla'
    model = clss.Classifier(params)
    # train and test
    model.train(train_files, train_pos, train_durations)
    nms_pos, nms_prob = model.test_batch(test_files, test_pos, test_durations, False, '')
    # compute precision recall
    precision, recall = evl.prec_recall_1d(nms_pos, nms_prob, test_pos, test_durations, model.params.detection_overlap, model.params.window_size)
    res.plot_prec_recall('rf', recall, precision, nms_prob)

    #
    # segment
    print '\nsegment'
    params.classification_model = 'segment'
    model = clss.Classifier(params)
    # train and test
    model.train(train_files, train_pos, train_durations)
    nms_pos, nms_prob = model.test_batch(test_files, test_pos, test_durations, False, '')
    # compute precision recall
    precision, recall = evl.prec_recall_1d(nms_pos, nms_prob, test_pos, test_durations, model.params.detection_overlap, model.params.window_size)
    res.plot_prec_recall('segment', recall, precision, nms_prob)

    #
    # scanr
    scanr_bat_results = base_line_dir + 'scanr/test_set_'+ test_set +'_scanr.csv'
    if os.path.isfile(scanr_bat_results):
        print '\nscanr'
        scanr_pos, scanr_prob = read_baseline_res(scanr_bat_results, test_files)
        precision_scanr, recall_scanr = evl.prec_recall_1d(scanr_pos, scanr_prob, test_pos, test_durations, params.detection_overlap, params.window_size)
        res.plot_prec_recall('scanr', recall_scanr, precision_scanr)

    #
    # sonobat
    sono_bat_results = base_line_dir + 'sonobat/test_set_'+ test_set +'_sono.csv'
    if os.path.isfile(sono_bat_results):
        print '\nsonobat'
        sono_pos, sono_prob = read_baseline_res(sono_bat_results, test_files)
        precision_sono, recall_sono = evl.prec_recall_1d(sono_pos, sono_prob, test_pos, test_durations, params.detection_overlap, params.window_size)
        res.plot_prec_recall('sonobat', recall_sono, precision_sono)

    #
    # kaleidoscope
    kal_bat_results = base_line_dir + 'kaleidoscope/test_set_'+ test_set +'_kaleidoscope.csv'
    if os.path.isfile(kal_bat_results):
        print '\nkaleidoscope'
        kal_pos, kal_prob = read_baseline_res(kal_bat_results, test_files)
        precision_kal, recall_kal = evl.prec_recall_1d(kal_pos, kal_prob, test_pos, test_durations, params.detection_overlap, params.window_size)
        res.plot_prec_recall('kaleidoscope', recall_kal, precision_kal)

    # save results
    plt.savefig(result_dir + test_set + '_results.png')
    plt.savefig(result_dir + test_set + '_results.pdf')
