import numpy as np
import evaluate as evl
import cls_audio_forest as cls_rf
import cls_cnn as cls_cnn
import cls_segment as seg
import create_results as res
import time


class Classifier:

    def __init__(self, params_):
        self.params = params_
        if self.params.classification_model == 'rf_vanilla':
            self.model = cls_rf.AudioForest(self.params)
        elif self.params.classification_model == 'cnn':
            self.model = cls_cnn.NeuralNet(self.params)
        elif self.params.classification_model == 'segment':
            self.model = seg.SegmentAudio(self.params)
        else:
            print 'Invalid model specified'

    def save_features(self, files):
        self.model.save_features(files)

    def train(self, files, gt_pos, durations):
        """
        Takes the file names and GT call positions and trains model.
        """

        positions, class_labels = generate_training_positions(files, gt_pos, durations, self.params)

        self.model.train(positions, class_labels, files, durations)

        # hard negative mining
        if self.params.num_hard_negative_mining > 0 and self.params.classification_model != 'segment':
            print '\nhard negative mining'
            for hn in range(self.params.num_hard_negative_mining):
                print '\thmn round', hn
                positions, class_labels = self.do_hnm(files, gt_pos, durations, positions, class_labels)
                self.model.train(positions, class_labels, files, durations)

    def test_single(self, audio_samples, sampling_rate):
        """
        Pass the raw audio samples and it will make a prediction.
        """
        duration = audio_samples.shape[0]/float(sampling_rate)
        nms_pos, nms_prob, y_prediction = self.model.test(file_duration=duration, audio_samples=audio_samples, sampling_rate=sampling_rate)
        return nms_pos, nms_prob, y_prediction

    def test_batch(self, files, gt_pos, durations, save_results=False, op_im_dir=''):
        """
        Takes a list of files as input and runs the detector on them.
        """
        nms_pos = [None]*len(files)
        nms_prob = [None]*len(files)
        for ii, file_name in enumerate(files):
            nms_pos[ii], nms_prob[ii], y_prediction = self.model.test(file_name=file_name, file_duration=durations[ii])

            # plot results
            if save_results:
                aud_file = self.params.audio_dir + file_name + '.wav'
                res.plot_spec(op_im_dir + file_name, aud_file, gt_pos[ii], nms_pos[ii], nms_prob[ii], y_prediction, self.params, True)

        return nms_pos, nms_prob

    def do_hnm(self, files, gt_pos, durations, positions, class_labels):
        """
        Hard negative mining, adds high confidence false positives to the training set.
        """

        nms_pos, nms_prob = self.test_batch(files, gt_pos, durations, False, '')

        positions_new = [None]*len(nms_pos)
        class_labels_new = [None]*len(nms_pos)
        for ii in range(len(files)):

            # add the false positives that are above the detection threshold
            # and not too close to the GT
            poss_negs = nms_pos[ii][nms_prob[ii][:,0] > self.params.detection_prob]
            if gt_pos[ii].shape[0] > 0:
                # have the extra newaxis in case gt_pos[ii] shape changes in the future
                pw_distance = np.abs(poss_negs[np.newaxis, ...]-gt_pos[ii][:,0][..., np.newaxis])
                dis_check = (pw_distance >= (self.params.window_size / 3)).mean(0)
                new_negs = poss_negs[dis_check==1]
            else:
                new_negs = poss_negs
            new_negs = new_negs[new_negs < (durations[ii]-self.params.window_size)]

            # add them to the training set
            positions_new[ii] = np.hstack((positions[ii], new_negs))
            class_labels_new[ii] = np.vstack((class_labels[ii], np.zeros((new_negs.shape[0], 1))))

            # sort
            sorted_inds = np.argsort(positions_new[ii])
            positions_new[ii] = positions_new[ii][sorted_inds]
            class_labels_new[ii] = class_labels_new[ii][sorted_inds]

        return positions_new, class_labels_new


def generate_training_positions(files, gt_pos, durations, params):
    positions = [None]*len(files)
    class_labels = [None]*len(files)
    for ii, ff in enumerate(files):
        positions[ii], class_labels[ii] = extract_train_position_from_file(gt_pos[ii], durations[ii], params)
    return positions, class_labels


def extract_train_position_from_file(gt_pos, duration, params):
    """
    Samples random negative locations for negs, making sure not to overlap with GT.

    gt_pos is the time in seconds that the call occurs.
    positions contains time in seconds of some negative and positive examples.
    """

    if gt_pos.shape[0] == 0:
        # dont extract any values if the file does not contain anything
        # we will use these ones for HNM later
        positions = np.zeros(0)
        class_labels = np.zeros((0,1))
    else:
        shift = 0  # if there is augmentation this is how much we will add
        num_neg_calls = gt_pos.shape[0]
        pos_window = params.window_size / 2  # window around GT that is not sampled from
        pos = gt_pos[:, 0]

        # augmentation
        if params.add_extra_calls:
            shift = params.aug_shift
            num_neg_calls *= 3
            pos = np.hstack((gt_pos[:, 0] - shift, gt_pos[:, 0], gt_pos[:, 0] + shift))

        # sample a set of negative locations - need to be sufficiently far away from GT
        pos_pad = np.hstack((0-params.window_size, gt_pos[:, 0], duration-params.window_size))
        neg = []
        cnt = 0
        while cnt < num_neg_calls:
            rand_pos = np.random.random()*pos_pad.max()
            if (np.abs(pos_pad - rand_pos) > (pos_window+shift)).mean() == 1:
                neg.append(rand_pos)
                cnt += 1
        neg = np.asarray(neg)

        # sort them
        positions = np.hstack((pos, neg))
        sorted_inds = np.argsort(positions)
        positions = positions[sorted_inds]

        # create labels
        class_labels = np.vstack((np.ones((pos.shape[0], 1)), np.zeros((neg.shape[0], 1))))
        class_labels = class_labels[sorted_inds]

    return positions, class_labels
