#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017   Chenglin Xu (NTU, Singapore)
# Updated by Chenglin, Dec 2018, Jul 2019

"""
1. Extract features (magnitude, log magnitude)
2. Converts to TFRecords format
3. Calculate global CMVN (same as kaldi).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing
import os,sys
import numpy as np
import tensorflow as tf

from utils.audioread import audioread
from utils.sigproc import framesig,magspec
from utils.normhamming import normhamming
import time

def make_sequence(feats, labels1=None, labels2=None):
    """
    Return a sequency for given feats and corresponding labels (optional for test)
    Args:
        feats: input feature vectors (i.e. magnitude of mixture speech)
        labels1: reference labels for target sepaker 1
        labels2: reference labels for target sepaker 2
    Returns:
        A tf.train.SequenceExample
    """

    inputs = [tf.train.Feature(float_list=tf.train.FloatList(value=feat)) for feat in feats]
    if labels1 is not None and labels2 is not None:
        targets1 = [tf.train.Feature(float_list=tf.train.FloatList(value=label)) for label in labels1]
        targets2 = [tf.train.Feature(float_list=tf.train.FloatList(value=label)) for label in labels2]
        feature_list = {
            'inputs': tf.train.FeatureList(feature=inputs),
            'labels1': tf.train.FeatureList(feature=targets1),
            'labels2': tf.train.FeatureList(feature=targets2)
        }
    else:
        feature_list = {
            'inputs': tf.train.FeatureList(feature=inputs)
        }

    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)

def cal_phase_mag(filename):
    '''
    extract phase and feats for one utterance
    '''
    
    rate, sig, _ = audioread(filename)
    frames = framesig(sig, FLAGS.FFT_LEN, FLAGS.FRAME_SHIFT, lambda x: normhamming(x), True)
    phase, feats = magspec(frames, FLAGS.FFT_LEN)

    return phase, feats

def extract_mag_feats(item, mix_dir, clean1_dir, clean2_dir, mean_var_dict):

    # tfrecords to save the sequency consisting of feats and labels (optional for test)
    tfrecords_name = os.path.join(FLAGS.output_dir, FLAGS.data_type, item.replace(".wav", ".tfrecords"))
    writer = tf.python_io.TFRecordWriter(tfrecords_name)

    # extract feats for mixture
    phase_mix, feats = cal_phase_mag(os.path.join(mix_dir, item))

    # calculate intermediates for mean and variance, save to kaldi vector format
    mean_feats = np.sum(feats, 0)
    var_feats = np.sum(np.square(feats), 0)
    mean_var_dict[item] = str(np.shape(feats)[0])+'+'+' '.join(str(mean_feat) for mean_feat in mean_feats)+'+'+' '.join(str(var_feat) for var_feat in var_feats)

    # extract mag for clean as labels
    if clean1_dir != '' and clean2_dir != '':
        phase_clean1, labels1 = cal_phase_mag(os.path.join(clean1_dir, item))
        phase_clean2, labels2 = cal_phase_mag(os.path.join(clean2_dir, item))

        if FLAGS.apply_psm:
            labels1 = labels1 * np.cos(phase_mix-phase_clean1)
            labels2 = labels2 * np.cos(phase_mix-phase_clean2)
    else:
        labels1 = None
        labels2 = None
    
    # write feats and labels into tfrecords
    writer.write(make_sequence(feats, labels1, labels2).SerializeToString())

    return mean_var_dict

def cal_global_mean_std(filename, mean_var_dict):
    cmvn = np.zeros((2, int(FLAGS.FFT_LEN/2+1)), dtype=np.float32)
    frames = 0.0
    for line in mean_var_dict:
        tokens = line.strip().split('+')
        frames += float(tokens[0])
        utt_mean_tokens = tokens[1].strip().split()
        cmvn[0] += [np.float32(i) for i in utt_mean_tokens]
        utt_var_tokens = tokens[2].strip().split()
        cmvn[1] += [np.float32(i) for i in utt_var_tokens]

    mean = cmvn[0] / frames
    var = cmvn[1] / frames - mean ** 2
    var[var<=0] = 1.0e-20
    std = np.sqrt(var)

    print(mean)
    print(std)
    np.savez(filename, mean_inputs=mean, stddev_inputs=std)

def main(unused_argv):
    print('Extract starts ...')
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))

    mix_dir = os.path.join(FLAGS.wav_dir, FLAGS.data_type, 'mix')
    if not os.path.exists(os.path.join(FLAGS.output_dir, FLAGS.data_type)):
        os.makedirs(os.path.join(FLAGS.output_dir, FLAGS.data_type))

    if FLAGS.with_labels:
        clean1_dir = os.path.join(FLAGS.wav_dir, FLAGS.data_type, 's1')
        clean2_dir = os.path.join(FLAGS.wav_dir, FLAGS.data_type, 's2')
    else:
        clean1_dir = ''
        clean2_dir = ''

    lists = [x for x in os.listdir(mix_dir) if x.endswith(".wav")]

    # check whether the cmvn file for training exist, remove if exist.
    if os.path.exists(FLAGS.inputs_cmvn):
        os.remove(FLAGS.inputs_cmvn)

    mean_vad_dict = multiprocessing.Manager().dict()
    pool = multiprocessing.Pool(FLAGS.num_threads)
    workers = []
    for item in lists:
        workers.append(pool.apply_async(extract_mag_feats(item, mix_dir, clean1_dir, clean2_dir, mean_vad_dict)))
    pool.close()
    pool.join()

    # convert the utterance level intermediates for mean and var to global mean and std, then save
    cal_global_mean_std(FLAGS.inputs_cmvn, mean_vad_dict.values())
    
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
    print('Extract ends.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--with_labels',
        type=int,
        default=1,
        help='Whether extract features for the targets as labels, default to prepare labels.')
    parser.add_argument(
        '--data_type',
        type=str,
        default='tr',
        help='tr, cv, tt.')
    parser.add_argument(
        '--apply_psm',
        type=int,
        default=1,
        help='Whether use phase sensitive mask.')
    parser.add_argument(
        '--inputs_cmvn',
        type=str,
        default='data/inputs_utts.cmvn',
        help='Path to save CMVN for the inputs'
    )
    parser.add_argument(
        '--wav_dir',
        type=str,
        default='data/wav',
        help='Directory to the input wav'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/tfrecords',
        help='Directory to save the features into tfrecords format'
    )
    parser.add_argument(
        '--FFT_LEN',
        type=int,
        default=512,
        help='The length of fft window.'
    )
    parser.add_argument(
        '--FRAME_SHIFT',
        type=int,
        default=256,
        help='The shift of samples when calculating fft.'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=10,
        help='The number of threads to convert tfrecords files.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
