#! /bin/bash

# Copyright 2017
# Author: Chenglin Xu (NTU, Singapore)
# Email: xuchenglin28@gmail.com
# Updated by Chenglin, Dec 2018, Jul 2019
# Please cite: 
#   Chenglin Xu, Wei Rao, Xiong Xiao, Eng Siong Chng and Haizhou Li, "SINGLE CHANNEL SPEECH SEPARATION WITH CONSTRAINED UTTERANCE LEVEL PERMUTATION INVARIANT TRAINING USING GRID LSTM", in Proc. of ICASSP 2018, pp 6-10.

step=1
gpu_id=$1 #'0', '1'

# Paths for reading wav and saving features
wav_dir=data/wsj0_2mix/wav8k/min
tfrecords_dir=data/tfrecords/pmag8k

# Configure for feature extraction
FFT_LEN=256 # 32ms for 8kHz sampling rate
FRAME_SHIFT=128 # 8ms for 8kHz sampling rate
with_labels=1
apply_psm=1

# Configure for network
rnn_num_layers=3
input_size=129
output_size=129
rnn_size=896
mask_type=relu # relu | sigmoid
dense_layer=false #true # set to false when use grid lstm, elsewise set true
tflstm_size=64 #0 # whether use grid lstm as first layer to replace dense_layer, set to 0 if you don't want to use grid lstm
tffeature_size=29
tffrequency_skip=10
tflstm_layers=1

# Configure for objective function
mag_factor=1.0
del_factor=4.5
acc_factor=10.0
dynamic_win=2

tPSA=0 # whether use non-negative phase sensitive mask
power_num=2 # mean square error (=2) or mean absolute error (=1)

# Configure for training
TF_CPP_MIN_LOG_LEVEL=1
tr_batch_size=2 # [change to situable size according to your GPU memory]
tt_batch_size=1
keep_prob=0.7
learning_rate=0.0005
lr_reduction_factor=0.7
min_epochs=30
max_epochs=200
num_threads=8

# Path to save model and reconstructed wav
prefix=PMag${mag_factor}_${tflstm_layers}_${tflstm_size}_${tffeature_size}_${tffrequency_skip}
name=${prefix}_BLSTM_${rnn_num_layers}_${rnn_size}_${mask_type}_del${del_factor}_acc${acc_factor}_win${dynamic_win}_P${power_num}
save_model_dir=exp/$name/

lists_dir=./tmp/psm_lists
mkdir -p $lists_dir

echo "FRAME_SHIFT=$FRAME_SHIFT" > config.py

#############
# use the code from http://www.merl.com/demos/deep-clustering to generate the wsj0_2mix data.
# only two speaker mixture without reverberation at 8kHz, with "min" length as the mixture length.
#############

# Prepare data for tr,cv
if [ $step -le 1 ]; then
    echo "Prepare data"
    for x in tr cv; do
        python extract_feats.py --data_type=$x --inputs_cmvn=$tfrecords_dir/${x}_cmvn.npz --with_labels=$with_labels --apply_psm=$apply_psm --wav_dir=$wav_dir --output_dir=$tfrecords_dir --FFT_LEN=$FFT_LEN --FRAME_SHIFT=$FRAME_SHIFT --num_threads=$num_threads &
    done
    wait
    echo "Prepare data done."
fi

# Training
if [ $step -le 2 ]; then
    echo "Model training starts."
    # sort the tfrecord files by size in order to group the utterances with similar length into a minibatch
    for x in tr cv; do
        ls -Sr $tfrecords_dir/${x}/*.tfrecords > $lists_dir/${x}.lst
    done
    command="python train.py --lists_dir=$lists_dir --save_model_dir=$save_model_dir --with_labels=$with_labels --dense_layer=$dense_layer \
        --rnn_num_layers=$rnn_num_layers --rnn_size=$rnn_size --tflstm_size=$tflstm_size --tffeature_size=$tffeature_size --tffrequency_skip=$tffrequency_skip \
        --tflstm_layers=$tflstm_layers --input_size=$input_size --output_size=$output_size --mask_type=$mask_type \
        --batch_size=$tr_batch_size --lr_reduction_factor=$lr_reduction_factor --learning_rate=$learning_rate --keep_prob=$keep_prob \
        --inputs_cmvn=$tfrecords_dir/tr_cmvn.npz --min_epochs=$min_epochs --max_epochs=$max_epochs --num_threads=$num_threads \
        --del_factor=$del_factor --acc_factor=$acc_factor --dynamic_win=$dynamic_win --mag_factor=$mag_factor --tPSA=$tPSA --power_num=$power_num "

    echo $command
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $command
    echo "Model training ends."
fi

# Prepare data for tr,cv,tt
if [ $step -le 3 ]; then
    echo "Prepare data"
    with_labels=0
    data_type=tt
    python extract_feats.py --data_type=$data_type --inputs_cmvn=$tfrecords_dir/${data_type}_cmvn.npz --with_labels=$with_labels --apply_psm=$apply_psm --wav_dir=$wav_dir --output_dir=$tfrecords_dir --FFT_LEN=$FFT_LEN --FRAME_SHIFT=$FRAME_SHIFT --num_threads=$num_threads
    echo "Prepare data done."
fi

# Decoding
if [ $step -le 4 ]; then
    echo "Decoding starts."

    # Configure for decoding
    with_labels=0 # set to 0 when the speakers in the mixture doesn't have corresponding clean labels
    data_type=tt
    noisy_dir=$wav_dir/$data_type/mix
    rec_dir=data/rec/$data_type/${name}

    ls -Sr $tfrecords_dir/${data_type}/*.tfrecords > $lists_dir/${data_type}.lst
    
    command="python decode.py --lists_dir=$lists_dir --data_type=$data_type --noisy_dir=$noisy_dir --save_model_dir=$save_model_dir --with_labels=$with_labels \
        --rec_dir=$rec_dir --dense_layer=$dense_layer --rnn_num_layers=$rnn_num_layers --rnn_size=$rnn_size --tflstm_size=$tflstm_size --tffeature_size=$tffeature_size \
        --tffrequency_skip=$tffrequency_skip --tflstm_layers=$tflstm_layers --input_size=$input_size --output_size=$output_size \
        --mask_type=$mask_type --keep_prob=$keep_prob --batch_size=$tt_batch_size --inputs_cmvn=$tfrecords_dir/tr_cmvn.npz \
        --num_threads=$num_threads --FFT_LEN=$FFT_LEN --FRAME_SHIFT=$FRAME_SHIFT --tPSA=$tPSA --power_num=$power_num "

    echo $command
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $command
    echo "Decoding ends."
fi

