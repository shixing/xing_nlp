#ls forders* | python read_hpc_output.py

import sys
import os
import pandas as pd


def process_file(path):
    f = open(os.path.join(path,'log.TRAIN.txt'))
    d = {}
    d['epoch'] = 1
    d['dev'] = 0.0
    d['train'] = 0.0
    d['_epoch'] = 1
    d['_dev'] = float("inf")
    d['_train'] = 0.0
    for line in f:
        if 'Dev_ppx' in line:
            ll = line.split()
            d['epoch'] = int(ll[1])
            d['dev'] = float(ll[-3])
            d['train'] = float(ll[-1])
            if d['_dev'] > d['dev']:
                d['_epoch'] = d['epoch']
                d['_dev'] = d['dev']
                d['_train'] = d['train']            

    f.close()
    return d
    

def main():

    table = {}

    for line in sys.stdin:
        folder = line.strip()
        row = process_file(folder)
        key = folder.split('/')[-1]
        table[key] = row

    # print the row
    df = pd.DataFrame.from_dict(table,orient = "index")
    pd.options.display.float_format = '{:.2f}'.format

    print df[['epoch','train','dev','_epoch','_train','_dev']]
    
if __name__ == '__main__':
    main()
