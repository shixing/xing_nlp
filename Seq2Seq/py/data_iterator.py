import numpy as np

PAD_ID = 0
START_ID = 1

class DataIterator:
    def __init__(self, model, data_set, n_bucket, batch_size, train_buckets_scale, data_order = None):
        self.data_set = data_set
        self.n_bucket = n_bucket
        self.batch_size = batch_size
        self.train_buckets_scale = train_buckets_scale
        self.model = model
        self.data_order = data_order

    def next_random(self):
        # first random bucket, then random sentences
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(self.train_buckets_scale))
                             if self.train_buckets_scale[i] > random_number_01])

            source_inputs, target_inputs, target_outputs, target_weights, _ = self.model.get_batch(self.data_set, bucket_id)
            yield source_inputs, target_inputs, target_outputs, target_weights, bucket_id

    def next_sequence(self, stop=False, test = False):
        # first select buckets from 0 to self.buckets-1, then select sentence one by one
        bucket_id = 0
        while True:
            if bucket_id >= self.n_bucket:
                if stop:
                    break
                bucket_id = 0
            start_id = 0
            while True:

                if test:
                    get_batch_func = self.model.get_batch_test

                    source_inputs, target_inputs, target_outputs, target_weights, finished = self.model.get_batch(self.data_set, bucket_id, start_id = start_id)
            
                    yield source_inputs, target_inputs, target_outputs, target_weights, bucket_id

                else:
                    get_batch_func = self.model.get_batch
                    source_inputs, target_inputs, target_outputs, target_weights, finished = self.model.get_batch(self.data_set, bucket_id, start_id = start_id)
 
                    yield source_inputs, target_inputs, target_outputs, target_weights, bucket_id

                if finished:
                    break
                start_id += self.batch_size

            bucket_id += 1


    def next_original(self):
        # according to original order
        # one by one
        
        for bucket_id, index in self.data_order:
            
            source_inputs, finished, length = self.model.get_batch_test(self.data_set, bucket_id, start_id = index)
 
            yield source_inputs, bucket_id, length

            
            
