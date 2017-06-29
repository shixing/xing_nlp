import nltk
import kenlm
import sys

def load_sentences():
    fn = "./sentences.txt"
    f = open(fn)
    line = f.read()
    words = nltk.tokenize.word_tokenize(line)
    words = [x.lower() for x in words]
    f.close()
    return words

def load_options():
    f = open("./options.txt")
    options = []
    for line in f: 
        ll = line.split()
        words = []
        for s in ll:
            word = s.split(".")[-1].lower()
            words.append(word)
        options.append(words)
    f.close()
    return options

def load_answers():
    f = open("./answers.txt")
    line = f.readline().strip()
    f.close()
    return line

def load_vocabs():
    f = open("../data/vocab.UNK")
    d = {}
    for line in f:
        line = line.strip()
        d[line] = 1
    f.close()
    return d

def unk_it(words, vocab):
    blank_ids = [str(n) for n in xrange(1,21)]
    for i in xrange(len(words)):
        if words[i] in blank_ids:
            continue
        if not words[i] in vocab:
            words[i] = "UNK"
    return words

def predict(words, options, lm, right_answers):
    ngram = lm.order
    blank_ids = [str(n) for n in xrange(1,21)]
    bid = 0
    answers = ""
    for i in xrange(len(words)):
        if words[i] in blank_ids:
            print "Blank ", words[i]
            
            max_score = -float('inf')
            choice = -1
            for j in xrange(len(options[bid])):
                option = options[bid][j]
                if not option in lm:
                    option = "UNK"
                seq = words[i - (ngram-1):i] + [option] + words[i+1:i + ngram]
                seq = " ".join(seq)
                score = lm.score(seq,bos=False, eos=False)
                print score, seq
                if score > max_score:
                    max_score = score
                    choice = j
            print "Choose: ", 'ABCD'[choice], "Correct: ", right_answers[bid]
            answers+='ABCD'[choice]
            bid += 1 

    return answers

def accuracy(answers,choices):
    n = len(answers)
    c = 0
    for i in xrange(len(answers)):
        if answers[i] == choices[i]:
            c += 1
    return c*1.0/n

def main():
    lm = kenlm.Model(sys.argv[1])

    words = load_sentences()
    options = load_options()
    answers = load_answers()

    print " ".join(words)

    words = unk_it(words, lm)
    print " ".join(words)
    
    
    choices = predict(words, options, lm, answers)
    print choices
    print answers
    
    a = accuracy(answers,choices)
    print a



if __name__ == "__main__":
    main()
        
