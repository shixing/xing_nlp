set PY=..\py_win\run.py
set MODEL_DIR=..\model\model_small
set TRAIN_PATH_FROM=..\data\small\train.src
set DEV_PATH_FROM=..\data\small\valid.src
set TEST_PATH_FROM=..\data\small\test.src
set TRAIN_PATH_TO=..\data\small\train.tgt
set DEV_PATH_TO=..\data\small\valid.tgt
set TEST_PATH_TO=..\data\small\test.tgt


set CUDA_VISIBLE_DEVICES=1


python %PY% --mode TRAIN --model_dir %MODEL_DIR% --train_path_from %TRAIN_PATH_FROM% --dev_path_from %DEV_PATH_FROM% --train_path_to %TRAIN_PATH_TO% --dev_path_to %DEV_PATH_TO% --batch_size 4 --from_vocab_size 100 --to_vocab_size 100 --size 100 --num_layers 2 --n_epoch 100 --saveCheckpoint True --attention False --learning_rate 0.1 --keep_prob 0.7

