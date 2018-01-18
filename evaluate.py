import numpy as np
from sklearn.metrics import roc_curve, auc


def compute_error_auc(op_str, gt, pred, prob):

    # classification error
    pred_int = (pred > prob).astype(np.int)
    class_acc = (pred_int == gt).mean() * 100.0

    # ROC - area under curve
    fpr, tpr, thresholds = roc_curve(gt, pred)
    roc_auc = auc(fpr, tpr)

    print op_str, ', class acc = %.3f, ROC AUC = %.3f' % (class_acc, roc_auc)
    #return class_acc, roc_auc


def calc_average_precision(recall, precision):

    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0

    # pascal'12 way
    mprec = np.hstack((0, precision, 0))
    mrec = np.hstack((0, recall, 1))
    for ii in range(mprec.shape[0]-2, -1,-1):
        mprec[ii] = np.maximum(mprec[ii], mprec[ii+1])
    inds = np.where(np.not_equal(mrec[1:], mrec[:-1]))[0]+1
    ave_prec = ((mrec[inds] - mrec[inds-1])*mprec[inds]).sum()

    return ave_prec


def remove_end_preds(nms_pos_o, nms_prob_o, gt_pos_o, durations, win_size):
    # this filters out predictions and gt that are close to the end
    # this is a bit messy because of the shapes of gt_pos_o
    nms_pos = []
    nms_prob = []
    gt_pos = []
    for ii in range(len(nms_pos_o)):
        valid_time = durations[ii] - win_size
        gt_cur = gt_pos_o[ii]
        if gt_cur.shape[0] > 0:
            gt_pos.append(gt_cur[:, 0][gt_cur[:, 0] < valid_time][..., np.newaxis])
        else:
            gt_pos.append(gt_cur)

        valid_preds = nms_pos_o[ii] < valid_time
        nms_pos.append(nms_pos_o[ii][valid_preds])
        nms_prob.append(nms_prob_o[ii][valid_preds, 0][..., np.newaxis])
    return nms_pos, nms_prob, gt_pos


def prec_recall_1d(nms_pos_o, nms_prob_o, gt_pos_o, durations, detection_overlap, win_size, remove_eof=True):
    """
    nms_pos, nms_prob, and gt_pos are lists of numpy arrays specifying detection
    position, detection probability and GT position.
    Each list entry is a different file.
    Each entry in nms_pos is an array of length num_entries. For nms_prob and
    gt_pos its an array of size (num_entries, 1).

    durations is a array of the length of the number of files with each entry
    containing that file length in seconds.
    detection_overlap determines if a prediction is counted as correct or not.
    win_size is used to ignore predictions and ground truth at the end of an
    audio file.

    returns
    precision: fraction of retrieved instances that are relevant.
    recall: fraction of relevant instances that are retrieved.
    """

    if remove_eof:
        # filter out the detections in both ground truth and predictions that are too
        # close to the end of the file - dont count them during eval
        nms_pos, nms_prob, gt_pos = remove_end_preds(nms_pos_o, nms_prob_o, gt_pos_o, durations, win_size)
    else:
        nms_pos = nms_pos_o
        nms_prob = nms_prob_o
        gt_pos = gt_pos_o

    # loop through each file
    true_pos = []  # correctly predicts the ground truth
    false_pos = []  # says there is a detection but isn't
    for ii in range(len(nms_pos)):
        num_preds = nms_pos[ii].shape[0]

        if num_preds > 0:  # check to make sure it contains something
            num_gt = gt_pos[ii].shape[0]

            # for each set of predictions label them as true positive or false positive (i.e. 1-tp)
            tp = np.zeros(num_preds)
            distance_to_gt = np.abs(gt_pos[ii].ravel()-nms_pos[ii].ravel()[:, np.newaxis])
            within_overlap = (distance_to_gt <= detection_overlap)

            # remove duplicate detections - assign to valid detection with highest prob
            for jj in range(num_gt):
                inds = np.where(within_overlap[:, jj])[0]  # get the indices of all valid predictions
                if inds.shape[0] > 0:
                    max_prob = np.argmax(nms_prob[ii][inds])
                    selected_pred = inds[max_prob]
                    within_overlap[selected_pred, :] = False
                    tp[selected_pred] = 1  # set as true positives
            true_pos.append(tp)
            false_pos.append(1 - tp)

    # calc precision and recall - sort confidence in descending order
    # PASCAL style
    conf = np.concatenate(nms_prob)[:, 0]
    num_gt = np.concatenate(gt_pos).shape[0]
    inds = np.argsort(conf)[::-1]
    true_pos_cat = np.concatenate(true_pos)[inds].astype(float)
    false_pos_cat = np.concatenate(false_pos)[inds].astype(float)  # i.e. 1-true_pos_cat

    if (conf == conf[0]).sum() == conf.shape[0]:
        # all the probability values are the same therefore we will not sweep
        # the curve and instead will return a single value
        true_pos_sum = true_pos_cat.sum()
        false_pos_sum = false_pos_cat.sum()

        recall = np.asarray([true_pos_sum / float(num_gt)])
        precision = np.asarray([(true_pos_sum / (false_pos_sum + true_pos_sum))])

    elif inds.shape[0] > 0:
        # otherwise produce a list of values
        true_pos_cum = np.cumsum(true_pos_cat)
        false_pos_cum = np.cumsum(false_pos_cat)

        recall = true_pos_cum / float(num_gt)
        precision = (true_pos_cum / (false_pos_cum + true_pos_cum))

    return precision, recall
