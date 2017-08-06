set PY=..\py_win\run.py
set MODEL_DIR=..\model\model_small_enfr_attr
set TRAIN_PATH_FROM=..\data\enfr\train_english.txt.tok.lc.10k
set DEV_PATH_FROM=..\data\enfr\dev_english.txt.tok.lc
set TEST_PATH_FROM=..\data\enfr\test_english.tok.lc
set TRAIN_PATH_TO=..\data\enfr\train_french.txt.tok.lc.10k
set DEV_PATH_TO=..\data\enfr\dev_french.txt.tok.lc
set TEST_PATH_TO=..\data\enfr\test_french.tok.lc


set CUDA_VISIBLE_DEVICES=1


python %PY% --mode TRAIN --model_dir %MODEL_DIR% --train_path_from %TRAIN_PATH_FROM% --dev_path_from %DEV_PATH_FROM% --train_path_to %TRAIN_PATH_TO% --dev_path_to %DEV_PATH_TO% --batch_size 4 --from_vocab_size 100 --to_vocab_size 100 --size 100 --num_layers 2 --n_epoch 100 --saveCheckpoint True --attention True --learning_rate 0.1 --keep_prob 0.7

