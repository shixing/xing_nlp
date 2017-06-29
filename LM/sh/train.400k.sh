lmplz -o 4 --skip_symbols "<unk>" < ../data/english.400k.txt > ../model/4gram400k.arpa.unk
build_binary ../model/4gram400k.arpa.unk ../model/4gram400k.binary.unk
