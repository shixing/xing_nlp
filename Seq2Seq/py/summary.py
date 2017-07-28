import tensorflow as tf



class ModelSummary:
    def __init__(self):
        with tf.name_scope("ModelSummary"):
            with tf.device("/cpu:0"):
                self.train_ppx = tf.placeholder(tf.float32, shape = (), name = "train_ppx")
                self.dev_ppx = tf.placeholder(tf.float32, shape = (), name = "dev_ppx")
                self.summary_train_ppx = tf.summary.scalar("train_ppx", self.train_ppx)
                self.summary_dev_ppx = tf.summary.scalar("dev_ppx", self.dev_ppx)
            
    def step_record(self, sess, train_ppx, dev_ppx):
        input_feed = {}
        input_feed[self.train_ppx.name] = train_ppx
        input_feed[self.dev_ppx.name] = dev_ppx
        
        output_feed = [self.summary_train_ppx, self.summary_dev_ppx ]
        
        outputs = sess.run(output_feed, input_feed)
        return outputs


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
