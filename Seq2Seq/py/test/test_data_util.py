import sys
sys.path.insert(0, "../")
import data_util
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(dir_path, "../../"))

def test_prepare_data():
    data_dir = os.path.join(root_dir, "data/ptb")
    train_path = os.path.join(data_dir, "train")
    dev_path = os.path.join(data_dir, "valid")
    vocab_size = 20000
    data_util.prepare_data(data_dir, train_path, dev_path, vocab_size)

def test_read_train_dev_test():
    data_dir = os.path.join(root_dir, "data/ptb")
    train_path = os.path.join(data_dir, "train")
    dev_path = os.path.join(data_dir, "valid")
    test_path = os.path.join(data_dir, "test")
    cache_dir = os.path.join(root_dir, "data/ptb/cache")
    vocab_size = 20000
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    train_data_bucket, dev_data_bucket, _buckets, vocab_path = data_util.read_train_dev(cache_dir, train_path, dev_path, vocab_size, 100, 10)
    test_data_bucket, _buckets_test = data_util.read_test(cache_dir, test_path, vocab_path, vocab_size, 100, 10)
    
    def print_bucket_data(data):
        l = [len(x) for x in data]
        print l

    print "_buckets: {}\n".format(_buckets)
    print_bucket_data(train_data_bucket)
    print_bucket_data(dev_data_bucket)
    print "_buckets_test: {}\n".format(_buckets_test)
    print_bucket_data(test_data_bucket)


def main():
    #test_prepare_data()
    test_read_train_dev_test()

if __name__ == "__main__":
    main()
