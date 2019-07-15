#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Chenglin Xu (NTU,Singapore)


"""Utility functions for computing dynamic features."""

import tensorflow as tf

def comp_dynamic_feature(inputs, DELTAWINDOW, Batch_size, lengths):
	outputs = []
	for i in range(Batch_size):
		tmp = comp_delta(inputs[i,:lengths[i],:], DELTAWINDOW, lengths[i])
		tmp1 = tf.pad(tmp, [[0,tf.reduce_max(lengths)-lengths[i]],[0,0]], "CONSTANT")
		outputs.append(tmp1)
	return tf.convert_to_tensor(outputs)

def comp_delta(feat, N, length):
	"""Compute delta features from a feature vector sequence.
	Args:
		feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
		N: For each frame, calculate delta features based on preceding and following N frames.
	Returns:
		A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    	"""
	feat = tf.concat([[feat[0] for i in range(N)], feat, [feat[-1] for i in range(N)]], 0)
	denom = sum([2*i*i for i in range(1,N+1)])
	dfeat = tf.reduce_sum([j*(feat[N+1+j-1:N+1+j+length-1,:]-feat[N+1-j-1:N+1-j+length-1,:]) for j in range(1,N+1)], axis=0)/denom
	return tf.convert_to_tensor(dfeat)
