PY=../py/run.py
BLEU=../py/multi-bleu.perl
MODEL_DIR=../model/model_small
TRAIN_PATH_FROM=../data/small/train.src
DEV_PATH_FROM=../data/small/valid.src
TEST_PATH_FROM=../data/small/test.src
TRAIN_PATH_TO=../data/small/train.tgt
DEV_PATH_TO=../data/small/valid.tgt
TEST_PATH_TO=../data/small/test.tgt
DECODE_OUTPUT=../data/small/test.output

perl $BLEU -lc $TEST_PATH_TO < $DECODE_OUTPUT
