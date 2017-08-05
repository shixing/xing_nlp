#!/bin/sh

#set -o nounset
#set -o errexit

#batch_size=32
#embedding_size=32
#learning_rate=0.56
#window_size=5

for batch_size in 24 32 48; do
for embedding_size in 16 32 48; do
for learning_rate in 0.49 0.52 0.62; do
for window_size in 2 3 5 6; do
train_params=e_${embedding_size}_b_${batch_size}_w_${window_size}_l_${learning_rate}
echo ${train_params}
python word2vec.py --train_data text8.small --eval_data questions-words.txt --save_path ./${train_params} --statistics_interval 2 --embedding_size $embedding_size --learning_rate $learning_rate --batch_size $batch_size --window_size $window_size 2>&1 | tee log_${train_params}.txt
done
done
done
done
