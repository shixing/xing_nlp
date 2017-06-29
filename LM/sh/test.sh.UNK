echo "5gram"
query -v summary ../model/5gram.binary.unk < ../data/ptb.valid.txt.UNK

echo "4gram"
query -v summary ../model/4gram.binary.unk < ../data/ptb.valid.txt.UNK

echo "3gram"
query -v summary ../model/3gram.binary.unk < ../data/ptb.valid.txt.UNK

echo "2gram"
query -v summary ../model/2gram.binary.unk < ../data/ptb.valid.txt.UNK

echo "3gram: i love you"
echo "i love you" | query ../model/3gram.binary.unk
