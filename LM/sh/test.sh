echo "3gram"
query -v summary ../model/3gram.binary < ../data/ptb.valid.txt

echo "2gram"
query -v summary ../model/2gram.binary < ../data/ptb.valid.txt

echo "3gram: i love you"
echo "i love you" | query ../model/3gram.binary
