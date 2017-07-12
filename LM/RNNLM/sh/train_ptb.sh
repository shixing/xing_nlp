PY=../py/run.py
MODEL_DIR=../model/model_ptb
TRAIN_PATH=../data/ptb/train
DEV_PATH=../data/ptb/valid
TEST_PATH=../data/ptb/test

source /home/nlg-05/xingshi/sh/init_tensorflow.sh

export CUDA_VISIBLE_DEVICES=1

python $PY --mode TRAIN --model_dir $MODEL_DIR \
    --train_path $TRAIN_PATH --dev_path $DEV_PATH \
    --batch_size 64 --vocab_size 10050 --size 300 --num_layers 2 \
    --n_epoch 40 --L 100 --n_bucket 10 --saveCheckpoint True --learning-rate 1.0 
