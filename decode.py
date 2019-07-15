#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017   Xu Chenglin(NTU, Singapore)
# Updated by Chenglin, Dec 2018, Jul 2019

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import pprint

import numpy as np
import tensorflow as tf
from model.model import Model
from utils.paddedFIFO_batch import paddedFIFO_batch
from utils.read_list import read_list

from utils.audioread import audioread
from utils.sigproc import framesig,magspec,deframesig
from utils.normhamming import normhamming
import scipy.io.wavfile as wav

FLAGS = None

def reconstruct(enhan_spec, noisy_file):

    rate, sig, nb_bits = audioread(noisy_file)
    frames = framesig(sig, FLAGS.FFT_LEN, FLAGS.FRAME_SHIFT, lambda x: normhamming(x), True)
    phase_noisy, _ = magspec(frames, FLAGS.FFT_LEN)
    
    spec_comp = enhan_spec * np.exp(phase_noisy * 1j)
    enhan_frames = np.fft.irfft(spec_comp)
    enhan_sig = deframesig(enhan_frames, len(sig), FLAGS.FFT_LEN, FLAGS.FRAME_SHIFT, lambda x: normhamming(x))
    enhan_sig = enhan_sig / np.max(np.abs(enhan_sig)) * np.max(np.abs(sig))
    enhan_sig = enhan_sig * float(2 ** (nb_bits - 1))
    if nb_bits == 16:
        enhan_sig = enhan_sig.astype(np.int16)
    elif nb_bits == 32:
        enhan_sig = enhan_sig.astype(np.int32)

    return enhan_sig, rate

def decode():
    tfrecords_list, num_batches = read_list(FLAGS.lists_dir, FLAGS.data_type, FLAGS.batch_size)

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            with tf.name_scope('input'):
                cmvn = np.load(FLAGS.inputs_cmvn)
                if FLAGS.with_labels:
                    inputs, inputs_cmvn, labels1, labels2, lengths = paddedFIFO_batch(tfrecords_list, FLAGS.batch_size,
                        FLAGS.input_size, FLAGS.output_size, cmvn=cmvn, with_labels=FLAGS.with_labels, 
                        num_enqueuing_threads=1, num_epochs=1, shuffle=False)
                else:
                    inputs, inputs_cmvn, lengths = paddedFIFO_batch(tfrecords_list, FLAGS.batch_size,
                        FLAGS.input_size, FLAGS.output_size, cmvn=cmvn, with_labels=FLAGS.with_labels, 
                        num_enqueuing_threads=1, num_epochs=1, shuffle=False)
                    labels1 = None
                    labels2 = None
               
        with tf.name_scope('model'):
            model = Model(FLAGS, inputs, inputs_cmvn, labels1, labels2, lengths, infer=True)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init)

        checkpoint = tf.train.get_checkpoint_state(FLAGS.save_model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            tf.logging.info("Restore best model from " + checkpoint.model_checkpoint_path)
            model.saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            tf.logging.fatal("Checkpoint is not found, please check the best model save path is correct.")
            sys.exit(-1)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
           for batch in xrange(num_batches):
               if coord.should_stop():
                   break
               
               sep1, sep2 = sess.run([model._sep1, model._sep2])

               filename = tfrecords_list[batch]
               (_, name) = os.path.split(filename)
               (uttid, _) = os.path.splitext(name)

               noisy_file = os.path.join(FLAGS.noisy_dir, uttid + '.wav')
               enhan_sig1, rate = reconstruct(np.squeeze(sep1), noisy_file)
               enhan_sig2, rate = reconstruct(np.squeeze(sep2), noisy_file)
               savepath1 = os.path.join(FLAGS.rec_dir, uttid + '_1.wav')
               savepath2 = os.path.join(FLAGS.rec_dir, uttid + '_2.wav')
               wav.write(savepath1, rate, enhan_sig1)
               wav.write(savepath2, rate, enhan_sig2)

               if (batch+1) % 300 == 0:
                   tf.logging.info("Number of batch processed: %d." % (batch+1))

        except Exception, e:
           coord.request_stop(e)
        finally:
           coord.request_stop()
           coord.join(threads)
        sess.close()

def main(_):
    if not os.path.exists(FLAGS.save_model_dir):
        tf.logging.fatal("The best model path is not exist, please check.")
        sys.exit(-1)

    if not os.path.exists(FLAGS.noisy_dir):
        tf.logging.fatal("The mixture speech path is not exist, please check. Use the phase of the mixture to reconstruct the separated speech.")
        sys.exit(-1)

    if not os.path.exists(FLAGS.rec_dir):
        os.makedirs(FLAGS.rec_dir)

    decode()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lists_dir',
        type=str,
        default='tmp/',
        help="List to show where the data is."
    )
    parser.add_argument(
        '--inputs_cmvn',
        type=str,
        default='tfrecords/tr_cmvn.npz',
        help="The global cmvn to normalize the inputs."
    )
    parser.add_argument(
        '--noisy_dir',
        type=str,
        default='min/tt/mixed',
        help="The directory where the mixture speech is."
    )
    parser.add_argument(
        '--data_type',
        type=str,
        default='tt',
        help="The data type to decode (default is tt, it's the folder name where the mixture speech is saved)."
    )
    parser.add_argument(
        '--rec_dir',
        type=str,
        default='data/wav/rec/model_name',
        help="The directory where the separated speech is saved."
    )
    parser.add_argument(
        '--with_labels',
        type=int,
        default=1,
        help='Whether the clean labels are included in the tfrecords.'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=129,
        help="Input feature dimension (default 129 for 8kHz sampling rate)."
    )
    parser.add_argument(
        '--output_size',
        type=int,
        default=129,
        help="Output dimension (mask dimension, default 129 for 8kHz sampling rate)."
    )
    parser.add_argument(
        '--dense_layer',
        type=str,
        default='false',
        help="Whether to use dense layer on top of input layer, when grid lstm is applied, this parameter is set to false, otherwise, set to true."
    )
    parser.add_argument(
        '--rnn_size',
        type=int,
        default=896,
        help="Number of units in a rnn layer."
    )
    parser.add_argument(
        '--rnn_num_layers',
        type=int,
        default=3,
        help="Number of rnn layers."
    )
    parser.add_argument(
        '--mask_type',
        type=str,
        default='relu',
        help="Mask avtivation funciton, now only support sigmoid or relu"
    )
    parser.add_argument(
        '--tflstm_size',
        type=int,
        default=64,
        help="unit size for grid lstm, 64"
    )
    parser.add_argument(
        '--tffeature_size',
        type=int,
        default=29,
        help="input size for the frequency dimension of grid lstm layer, 29"
    )
    parser.add_argument(
        '--tffrequency_skip',
        type=int,
        default=10,
        help="shift of the input for the frequency dimension of grid lstm layer, 10"
    )
    parser.add_argument(
        '--tflstm_layers',
        type=int,
        default=1,
        help="number of grid lstm layers, 1"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help="Minibatch size."
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=12,
        help='Number of threads for paralleling.'
    )
    parser.add_argument(
        '--save_model_dir',
        type=str,
        default='exp/model_name',
        help="Directory to save the training model in every epoch."
    )
    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.5,
        help="Keep probability for training with a dropout (default: 1-dropout_rate)."
    )
    parser.add_argument(
        '--tPSA',
        type=int,
        default=0,
        help="Whether use truncted PSA."
    )
    parser.add_argument(
        '--power_num',
        type=int,
        default=2,
        help="The power to calculate the loss, if set to 2, it's squared L2, if set to 1, it's L1."
    )
    parser.add_argument(
        '--FFT_LEN',
        type=int,
        default=256,
        help="The length of FFT."
    )
    parser.add_argument(
        '--FRAME_SHIFT',
        type=int,
        default=64,
        help="The frame shift."
    )
    FLAGS, unparsed = parser.parse_known_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__dict__)
    sys.stdout.flush()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
