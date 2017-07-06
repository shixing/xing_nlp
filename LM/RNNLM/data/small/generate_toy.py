import sys
import os
import random

def generate(path, nline, max_length):
    f = open(path, 'w')
    for i in xrange(nline):
        length = random.randint(1, max_length)
        words = []
        for j in xrange(length):
            c  = chr(97 + random.randint(0,25))
            words.append(c)
        f.write("{}\n".format(" ".join(words)))
    f.close()
    
ns = [1000,100,100]
max_length = 20
generate("train",ns[0],max_length)
generate("valid",ns[1],max_length)
generate("test",ns[2],max_length)
