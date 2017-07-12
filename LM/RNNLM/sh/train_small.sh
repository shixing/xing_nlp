PY=../py/run.py
MODEL_DIR=../model/model_small
TRAIN_PATH=../data/small/train
DEV_PATH=../data/small/valid
TEST_PATH=../data/small/test

export CUDA_VISIBLE_DEVICES=1


python $PY --mode TRAIN --model_dir $MODEL_DIR \
    --train_path $TRAIN_PATH --dev_path $DEV_PATH \
    --batch_size 4 --vocab_size 100 --size 20 --num_layers 2 \
    --n_epoch 100 --L 15 --n_bucket 3 --saveCheckpoint True
    
