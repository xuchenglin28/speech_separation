#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017  Xu Chenglin(NTU, Singapore)
# Updated by Chenglin, Dec 2018, Jul 2019

"""
1. Build speech separation network structure
2. Calculate the objective loss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn
from utils.comp_dynamic_feature import comp_dynamic_feature

class Model(object):

    def __init__(self, config, inputs, inputs_norm, labels1=None, labels2=None, lengths=None, infer=False):
        self._config = config
        self._mixed = inputs
        self._inputs = inputs_norm
        if labels1 is not None and labels2 is not None:
            if self._config.tPSA:
                self._labels1 = self.get_tPSA(labels1)
                self._labels2 = self.get_tPSA(labels2)
            else:
                self._labels1 = labels1
                self._labels2 = labels2
        self._lengths = lengths
        self._infer = infer

        self.build_model()

    def get_tPSA(self, labels):
        return tf.minimum(tf.maximum(labels, tf.constant(0, dtype=labels.dtype)), self._mixed)

    def build_model(self):
        self.build_net()
        if self._infer: return
        self.cal_loss()
        if tf.get_variable_scope().reuse: return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars), self._config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def build_net(self):
        outputs = self._inputs
        # feed-forward layer, not used (set to false) when grid lstm is applied
        if self._config.dense_layer.lower() == 'true':
            with tf.variable_scope('forward1'):
                outputs = tf.reshape(outputs, [-1, self._config.input_size])
                outputs = tf.layers.dense(outputs, units=self._config.rnn_size,
                              activation=tf.nn.tanh, reuse=tf.get_variable_scope().reuse)
                outputs = tf.reshape(outputs, [self._config.batch_size, -1, self._config.rnn_size])
        
        # grid lstm layer and a linear reduction layer
        if self._config.tflstm_size > 0:
            with tf.variable_scope('tflstm'):
                def tflstm_cell():
                    return tf.contrib.rnn.GridLSTMCell(self._config.tflstm_size, use_peepholes=True, share_time_frequency_weights=True, 
                               cell_clip=5.0, feature_size=self._config.tffeature_size, frequency_skip=self._config.tffrequency_skip, 
                               num_frequency_blocks=[int((self._config.input_size-self._config.tffeature_size)/self._config.tffrequency_skip+1)])
                    
                cell = tf.contrib.rnn.MultiRNNCell([tflstm_cell() for _ in range(self._config.tflstm_layers)], state_is_tuple=True)
                initial_state = cell.zero_state(self._config.batch_size, tf.float32)
                outputs, final_state = tf.nn.dynamic_rnn(cell, outputs, dtype=tf.float32, sequence_length=self._lengths, initial_state=initial_state)

                tflstm_output_size = 2*self._config.tflstm_size*int((self._config.input_size-self._config.tffeature_size)/self._config.tffrequency_skip+1)
                outputs = tf.reshape(outputs, [-1, tflstm_output_size])
                weights, biases = self._weight_and_bias('linear', tflstm_output_size, self._config.rnn_size)
                outputs = tf.matmul(outputs, weights) + biases
                outputs = tf.reshape(outputs, [self._config.batch_size, -1, self._config.rnn_size])

        # BLSTM layer
        with tf.variable_scope('blstm'):
            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(self._config.rnn_size) #tf.nn.rnn_cell.BasicLSTMCell in r1.12
            attn_cell = lstm_cell
            if not self._infer and self._config.keep_prob < 1.0:
                def attn_cell():
                    return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self._config.keep_prob)

            # tf.nn.rnn_cell.MultiRNNCell in r1.12
            lstm_fw_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self._config.rnn_num_layers)], state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self._config.rnn_num_layers)], state_is_tuple=True)
            lstm_fw_cell = self._unpack_cell(lstm_fw_cell)
            lstm_bw_cell = self._unpack_cell(lstm_bw_cell)
            outputs, fw_final_states, bw_final_states = rnn.stack_bidirectional_dynamic_rnn(cells_fw=lstm_fw_cell,
                cells_bw=lstm_bw_cell, inputs=outputs, dtype=tf.float32, sequence_length=self._lengths)

        # Mask estimation layer
        with tf.variable_scope('forward2'):
            blstm_output_size = 2*self._config.rnn_size
            outputs = tf.reshape(outputs, [-1, blstm_output_size])
            
            weights1, biases1 = self._weight_and_bias('mask1', blstm_output_size, self._config.output_size)
            weights2, biases2 = self._weight_and_bias('mask2', blstm_output_size, self._config.output_size)
            if self._config.mask_type.lower() == 'relu':
                mask1 = tf.nn.relu(tf.matmul(outputs, weights1) + biases1)
                mask2 = tf.nn.relu(tf.matmul(outputs, weights2) + biases2)
            else:
                mask1 = tf.nn.sigmoid(tf.matmul(outputs, weights1) + biases1)
                mask2 = tf.nn.sigmoid(tf.matmul(outputs, weights2) + biases2)
            
            self._mask1 = tf.reshape(mask1, [self._config.batch_size, -1, self._config.output_size])
            self._mask2 = tf.reshape(mask2, [self._config.batch_size, -1, self._config.output_size])

            self._sep1 = self._mask1 * self._mixed
            self._sep2 = self._mask2 * self._mixed

        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)

    def cal_loss(self):
        if self._config.mag_factor > 0.0:
            cost1 = tf.reduce_sum(tf.reduce_sum(tf.abs(tf.pow(self._sep1 - self._labels1, self._config.power_num)), 1) +
                        tf.reduce_sum(tf.abs(tf.pow(self._sep2 - self._labels2, self._config.power_num)), 1), 1)
            cost2 = tf.reduce_sum(tf.reduce_sum(tf.abs(tf.pow(self._sep2 - self._labels1, self._config.power_num)), 1) +
                        tf.reduce_sum(tf.abs(tf.pow(self._sep1 - self._labels2, self._config.power_num)), 1), 1)
            cost1 = tf.multiply(self._config.mag_factor, cost1)
            cost2 = tf.multiply(self._config.mag_factor, cost2)
        else:
            cost1 = 0.0
            cost2 = 0.0

        if self._config.del_factor > 0.0:
            sep_delta1 = comp_dynamic_feature(self._sep1, self._config.dynamic_win, self._config.batch_size, self._lengths)
            sep_delta2 = comp_dynamic_feature(self._sep2, self._config.dynamic_win, self._config.batch_size, self._lengths)
            labels_delta1 = comp_dynamic_feature(self._labels1, self._config.dynamic_win, self._config.batch_size, self._lengths)
            labels_delta2 = comp_dynamic_feature(self._labels2, self._config.dynamic_win, self._config.batch_size, self._lengths)
            cost_del1 = tf.reduce_sum(tf.reduce_sum(tf.abs(tf.pow(sep_delta1 - labels_delta1, self._config.power_num)), 1) +
                            tf.reduce_sum(tf.abs(tf.pow(sep_delta2 - labels_delta2, self._config.power_num)), 1), 1)
            cost_del2 = tf.reduce_sum(tf.reduce_sum(tf.abs(tf.pow(sep_delta2 - labels_delta1, self._config.power_num)), 1) +
                            tf.reduce_sum(tf.abs(tf.pow(sep_delta1 - labels_delta2, self._config.power_num)), 1), 1)

            cost1 += tf.multiply(self._config.del_factor, cost_del1)
            cost2 += tf.multiply(self._config.del_factor, cost_del2)

        if self._config.acc_factor > 0.0:
            sep_acc1 = comp_dynamic_feature(sep_delta1, self._config.dynamic_win, self._config.batch_size, self._lengths)
            sep_acc2 = comp_dynamic_feature(sep_delta2, self._config.dynamic_win, self._config.batch_size, self._lengths)
            labels_acc1 = comp_dynamic_feature(labels_delta1, self._config.dynamic_win, self._config.batch_size, self._lengths)
            labels_acc2 = comp_dynamic_feature(labels_delta2, self._config.dynamic_win, self._config.batch_size, self._lengths)
            cost_acc1 = tf.reduce_sum(tf.reduce_sum(tf.abs(tf.pow(sep_acc1 - labels_acc1, self._config.power_num)), 1) +
                            tf.reduce_sum(tf.abs(tf.pow(sep_acc2 - labels_acc2, self._config.power_num)), 1), 1)
            cost_acc2 = tf.reduce_sum(tf.reduce_sum(tf.abs(tf.pow(sep_acc2 - labels_acc1, self._config.power_num)), 1) +
                            tf.reduce_sum(tf.abs(tf.pow(sep_acc1 - labels_acc2, self._config.power_num)), 1), 1)

            cost1 += tf.multiply(self._config.acc_factor, cost_acc1)
            cost2 += tf.multiply(self._config.acc_factor, cost_acc2)

        cost1 = tf.div(cost1, tf.to_float(self._lengths))
        cost2 = tf.div(cost2, tf.to_float(self._lengths))
        
        # find the optimal permuration and obtain the minimum loss
        idx = tf.cast(cost1 > cost2, tf.float32)
        self._loss = tf.reduce_sum(idx * cost2 + (1 - idx) * cost1)
    
    def _weight_and_bias(self, scope_name, input_size, output_size):
        weights = tf.get_variable(scope_name+'_weights', [input_size, output_size], initializer=tf.random_normal_initializer(stddev=0.01))
        biases = tf.get_variable(scope_name+'_biases', [output_size], initializer=tf.constant_initializer(0.0))
        return weights, biases

    def _unpack_cell(self, cell):
        if isinstance(cell,tf.contrib.rnn.MultiRNNCell):
            return cell._cells
        else:
            return [cell]
