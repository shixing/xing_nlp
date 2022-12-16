from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import re
import tarfile

from tensorflow.python.platform import gfile
import tensorflow as tf

from best_buckets import calculate_buckets, split_buckets, get_buckets_id



# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"


def maybe_download(directory, filename, url):
    """Download filename from url unless it's already in directory."""
    if not os.path.exists(directory):
        print("Creating directory %s" % directory)
        os.mkdir(directory)
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        print("Downloading %s to %s" % (url, filepath))
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        statinfo = os.stat(filepath)
        print("Successfully downloaded", filename, statinfo.st_size, "bytes")
    return filepath


def gunzip_file(gz_path, new_path):
    """Unzips from gz_path into new_path."""
    print("Unpacking %s to %s" % (gz_path, new_path))
    with gzip.open(gz_path, "rb") as gz_file:
        with open(new_path, "wb") as new_file:
            for line in gz_file:
                new_file.write(line)


def get_wmt_enfr_train_set(directory):
    """Download the WMT en-fr training corpus to directory unless it's there."""
    train_path = os.path.join(directory, "giga-fren.release2.fixed")
    if not (gfile.Exists(train_path +".fr") and gfile.Exists(train_path +".en")):
        corpus_file = maybe_download(directory, "training-giga-fren.tar",
                                                                  _WMT_ENFR_TRAIN_URL)
        print("Extracting tar file %s" % corpus_file)
        with tarfile.open(corpus_file, "r") as corpus_tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(corpus_tar, directory)
        gunzip_file(train_path + ".fr.gz", train_path + ".fr")
        gunzip_file(train_path + ".en.gz", train_path + ".en")
    return train_path


def get_wmt_enfr_dev_set(directory):
    """Download the WMT en-fr training corpus to directory unless it's there."""
    dev_name = "newstest2013"
    dev_path = os.path.join(directory, dev_name)
    if not (gfile.Exists(dev_path + ".fr") and gfile.Exists(dev_path + ".en")):
        dev_file = maybe_download(directory, "dev-v2.tgz", _WMT_ENFR_DEV_URL)
        print("Extracting tgz file %s" % dev_file)
        with tarfile.open(dev_file, "r:gz") as dev_tar:
            fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
            en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
            fr_dev_file.name = dev_name + ".fr"  # Extract without "dev/" prefix.
            en_dev_file.name = dev_name + ".en"
            dev_tar.extract(fr_dev_file, directory)
            dev_tar.extract(en_dev_file, directory)
    return dev_path


def blank_tokenizer(sentence):
    return sentence.strip().split()

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_paths, max_vocabulary_size,
                                            tokenizer=None, normalize_digits=False):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
        vocabulary_path: path where the vocabulary will be created.
        data_path: data file that will be used to create vocabulary.
        max_vocabulary_size: limit on the size of the created vocabulary.
        tokenizer: a function to use to tokenize each data sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, ",".join(data_paths)))
        vocab = {}
        for data_path in data_paths:
            with gfile.GFile(data_path, mode="rb") as f:
                print(data_path)
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  processing line %d" % counter)
                    line = tf.compat.as_bytes(line)
                    tokens = tokenizer(line) if tokenizer else blank_tokenizer(line)
                    for w in tokens:
                        word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1
                print(len(vocab))
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
        dog
        cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
        vocabulary_path: path to the file containing the vocabulary.

    Returns:
        a pair: the vocabulary (a dictionary mapping string to integers), and
        the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
        ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                                                    tokenizer=None, normalize_digits=False, with_start = True, with_end = True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
        sentence: the sentence in bytes format to convert to token-ids.
        vocabulary: a dictionary mapping tokens to integers.
        tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
        a list of integers, the token-ids for the sentence.
    """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        ids =  [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    else:
        ids =  [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]
    if with_start:
            ids = [GO_ID] + ids
    if with_end:
            ids =  ids + [EOS_ID]
    return ids

def data_to_token_ids(data_path, target_path, vocabulary_path,
                                            tokenizer=None, normalize_digits=False, with_go = True, with_end = True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
        data_path: path to the data file in one-sentence-per-line format.
        target_path: path where the file with token-ids will be created.
        vocabulary_path: path to the vocabulary file.
        tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,tokenizer, normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")



def prepare_data(cache_dir, train_path, dev_path, vocabulary_size):
    """Preapre all necessary files that are required for the training.

        Args:
            data_dir: directory in which the data sets will be stored.
            all the sentence already prepend _GO and append _EOS

    """
    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(cache_dir, "vocab")
    create_vocabulary(vocab_path, [train_path, dev_path], vocabulary_size)

    # Create token ids for the training data.
    train_ids_path =  os.path.join(cache_dir, "train.ids")
    data_to_token_ids(train_path, train_ids_path, vocab_path)

    # Create token ids for the development data.
    dev_ids_path = os.path.join(cache_dir, "dev.ids")
    data_to_token_ids(dev_path, dev_ids_path, vocab_path)

    return train_ids_path, dev_ids_path, vocab_path






def read_raw_data(target_path, max_size=None):
    '''
    Args: 
        target_path : the path which contains word ids
    '''
    print("read raw data from {}".format(target_path))
    data_set = []
    data_length = []

    with tf.gfile.GFile(target_path, mode="r") as target_file:
        target = target_file.readline()
        counter = 0
        while target and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            target_ids = [int(x) for x in target.split()]
            data_set.append(target_ids)
            data_length.append(len(target_ids))
            target = target_file.readline()


    return data_set, data_length


def get_vocab_path(cache_dir):
    vocab_path = os.path.join(cache_dir, "vocab")
    return vocab_path

def get_real_vocab_size(vocab_path):
    n = 0
    f = open(vocab_path)
    for line in f:
        n+=1
    f.close()
    return n
    

def read_train_dev(cache_dir, train_path, dev_path, vocab_size, max_length, n_bucket):
    train_ids_path, dev_ids_path, vocab_path  = prepare_data(cache_dir, train_path, dev_path, vocab_size)
    train_data, train_length = read_raw_data(train_ids_path)
    dev_data, dev_length = read_raw_data(dev_ids_path)
    length_array = train_length + dev_length
    _buckets = calculate_buckets(length_array, max_length, n_bucket)
    train_data_bucket,_ = split_buckets(train_data, _buckets)
    dev_data_bucket,_ = split_buckets(dev_data, _buckets)
    return train_data_bucket, dev_data_bucket, _buckets, vocab_path

def read_test(cache_dir, test_path, vocab_path, max_length, n_bucket):
    global _buckets
    test_ids_path = os.path.join(cache_dir, "test.ids")
    data_to_token_ids(test_path, test_ids_path, vocab_path)
    test_data, test_length = read_raw_data(test_ids_path)
    _buckets = calculate_buckets(test_length, max_length, n_bucket)
    test_data_bucket, test_data_order = split_buckets(test_data, _buckets)    
    return test_data_bucket, _buckets, test_data_order

