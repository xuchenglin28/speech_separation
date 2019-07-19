# Constrained Permutation Invariant Training, Speech Separation

Please cite:

 1. Chenglin Xu, Wei Rao, Xiong Xiao, Eng Siong Chng and Haizhou Li, "SINGLE CHANNEL SPEECH SEPARATION WITH CONSTRAINED UTTERANCE LEVEL PERMUTATION INVARIANT TRAINING USING GRID LSTM", in Proc. of ICASSP 2018, pp 6-10.
 
 2. Chenglin Xu, Wei Rao, Eng Siong Chng and Haizhou Li, "A Shifted Delta Coefficient Objective for Monaural Speech Separation using Multi-task Learning", in Proc. of INTERSPEECH 2018, pp 3479-3483.

## Data Generation

If you are using wsj0-2mix to repeat the work in paper 1 and 2, please use the code from http://www.merl.com/demos/deep-clustering to generate the wsj0_2mix data

## Speech Separation

Currently, the code only implement two speaker separation, if you have more speakers to be separated, please revise the output part together with mask estimation accordingly. The number of speakers information need to be known in prior, it limits the application of speech separation in practice.

We have done another work, which is target speaker extraction. It's only extracting target speaker's voice from the mixed or noisy enviroment. please refer to https://github.com/xuchenglin28/speaker_extraction. 

The run.sh script includes feature extraction, modeling training and run-time inference, please run it after you simulated data.

run.sh

## Environments:

python: 2.7

Tensorflow: 1.12 (some API are older version, but compatiable by 1.12)

Part of the code are adapted from https://github.com/snsun/pit-speech-separation

## Contact:

email: xuchenglin28@gmail.com
