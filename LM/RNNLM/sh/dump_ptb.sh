PY=../py/run.py
MODEL_DIR=../model/model_ptb
TRAIN_PATH=../data/ptb/train
DEV_PATH=../data/ptb/valid
TEST_PATH=../data/ptb/test

source /home/nlg-05/xingshi/sh/init_tensorflow.sh

export CUDA_VISIBLE_DEVICES=1

python $PY --mode DUMP_LSTM --model_dir $MODEL_DIR \
    --test_path $TEST_PATH --size 300 --num_layers 2 \
    --L 100 --n_bucket 10
