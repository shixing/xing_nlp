lmplz -o 3 --skip_symbols "<unk>" < ../data/ptb.train.txt > ../model/3gram.arpa
build_binary ../model/3gram.arpa ../model/3gram.binary

lmplz -o 2 --skip_symbols "<unk>" < ../data/ptb.train.txt > ../model/2gram.arpa
build_binary ../model/2gram.arpa ../model/2gram.binary

lmplz -o 1 --skip_symbols "<unk>" < ../data/ptb.train.txt > ../model/1gram.arpa
build_binary ../model/1gram.arpa ../model/1gram.binary




