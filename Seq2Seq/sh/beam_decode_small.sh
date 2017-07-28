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

export CUDA_VISIBLE_DEVICES=1


python $PY --mode BEAM_DECODE --model_dir $MODEL_DIR \
    --test_path_from $TEST_PATH_FROM \
    --beam_size 10 --from_vocab_size 100 --to_vocab_size 100 --size 100 --num_layers 2 \
    --attention False --print_beam True --decode_output $DECODE_OUTPUT

