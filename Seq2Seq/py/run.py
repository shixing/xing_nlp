from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import logging
import data_utils
from seqModel import SeqModel

import data_iterator
from data_iterator import DataIterator
from tensorflow.python.client import timeline

from summary import ModelSummary, variable_summaries

from google.protobuf import text_format

from state import StateWrapper


############################
######## MARK:FLAGS ########
############################

# mode
tf.app.flags.DEFINE_string("mode", "TRAIN", "TRAIN|FORCE_DECODE|BEAM_DECODE|DUMP_LSTM")

# datasets, paths, and preprocessing
tf.app.flags.DEFINE_string("model_dir", "./model", "model_dir/data_cache/n model_dir/saved_model; model_dir/log.txt .")
tf.app.flags.DEFINE_string("train_path_from", "./train", "the absolute path of raw source train file.")
tf.app.flags.DEFINE_string("dev_path_from", "./dev", "the absolute path of raw source dev file.")
tf.app.flags.DEFINE_string("test_path_from", "./test", "the absolute path of raw source test file.")

tf.app.flags.DEFINE_string("train_path_to", "./train", "the absolute path of raw target train file.")
tf.app.flags.DEFINE_string("dev_path_to", "./dev", "the absolute path of raw target dev file.")
tf.app.flags.DEFINE_string("test_path_to", "./test", "the absolute path of raw target test file.")

tf.app.flags.DEFINE_string("decode_output", "./output", "beam search decode output.")


tf.app.flags.DEFINE_string("force_decode_output", "force_decode.txt", "the file name of the score file as the output of force_decode. The file will be put at model_dir/force_decode_output")
tf.app.flags.DEFINE_string("dump_lstm_output", "dump_lstm.pb", "the file to save hidden states as a protobuffer as the output of dump_lstm. The file will be put at model_dir/dump_lstm_output")



# tuning hypers
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.83,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "dropout rate.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training/evaluation.")

tf.app.flags.DEFINE_integer("from_vocab_size", 10000, "from vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 10000, "to vocabulary size.")

tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("n_epoch", 500,
                            "Maximum number of epochs in training.")

# replaced by the bucket size
# tf.app.flags.DEFINE_integer("L", 30,"max length")

tf.app.flags.DEFINE_integer("n_bucket", 10,
                            "num of buckets to run.")
tf.app.flags.DEFINE_integer("patience", 10,"exit if the model can't improve for $patence evals")

# devices
tf.app.flags.DEFINE_string("N", "000", "GPU layer distribution: [input_embedding, lstm, output_embedding]")

# training parameter
tf.app.flags.DEFINE_boolean("withAdagrad", True,
                            "withAdagrad.")
tf.app.flags.DEFINE_boolean("fromScratch", True,
                            "withAdagrad.")
tf.app.flags.DEFINE_boolean("saveCheckpoint", False,
                            "save Model at each checkpoint.")
tf.app.flags.DEFINE_boolean("profile", False, "False = no profile, True = profile")

# for beam_decode
tf.app.flags.DEFINE_integer("beam_size", 10,"the beam size")
tf.app.flags.DEFINE_boolean("print_beam", False, "to print beam info")
tf.app.flags.DEFINE_float("min_ratio", 0.5, "min_ratio.")
tf.app.flags.DEFINE_float("max_ratio", 1.5, "max_ratio.")


# GPU configuration
tf.app.flags.DEFINE_boolean("allow_growth", False, "allow growth")

# Summary
tf.app.flags.DEFINE_boolean("with_summary", False, "with_summary")

# With Attention
tf.app.flags.DEFINE_boolean("attention", False, "with_attention")



FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
_buckets = [(10, 10), (22, 22)]
_beam_buckets = [10, 22]


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.
  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()][::-1]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def read_data_test(source_path):

    order = []
    data_set = [[] for _ in _beam_buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        source = source_file.readline()
        counter = 0
        while source:
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in source.split()][::-1]
            for bucket_id, source_size in enumerate(_beam_buckets):
                if len(source_ids) < source_size:

                    order.append((bucket_id, len(data_set[bucket_id])))
                    data_set[bucket_id].append(source_ids)
                    
                    break
            source = source_file.readline()
    return data_set, order



def mylog(msg):
    print(msg)
    sys.stdout.flush()
    logging.info(msg)


def mylog_section(section_name):
    mylog("======== {} ========".format(section_name)) 

def mylog_subsection(section_name):
    mylog("-------- {} --------".format(section_name)) 

def mylog_line(section_name, message):
    mylog("[{}] {}".format(section_name, message))


def get_device_address(s):
    add = []
    if s == "":
        for i in xrange(3):
            add.append("/cpu:0")
    else:
        add = ["/gpu:{}".format(int(x)) for x in s]

    return add


def dump_graph(fn):
    graph = tf.get_default_graph()
    graphDef = graph.as_graph_def()
        
    text = text_format.MessageToString(graphDef)
    f = open(fn,'w')
    f.write(text)
    f.close()

def show_all_variables():
    all_vars = tf.global_variables()
    for var in all_vars:
        mylog(var.name)


def log_flags():
    members = FLAGS.__dict__['__flags'].keys()
    mylog_section("FLAGS")
    for attr in members:
        mylog("{}={}".format(attr, getattr(FLAGS, attr)))


def create_model(session, run_options, run_metadata):
    devices = get_device_address(FLAGS.N)
    dtype = tf.float32
    model = SeqModel(FLAGS._buckets,
                     FLAGS.size,
                     FLAGS.real_vocab_size_from,
                     FLAGS.real_vocab_size_to,
                     FLAGS.num_layers,
                     FLAGS.max_gradient_norm,
                     FLAGS.batch_size,
                     FLAGS.learning_rate,
                     FLAGS.learning_rate_decay_factor,
                     withAdagrad = FLAGS.withAdagrad,
                     dropoutRate = FLAGS.keep_prob,
                     dtype = dtype,
                     devices = devices,
                     topk_n = FLAGS.beam_size,
                     run_options = run_options,
                     run_metadata = run_metadata,
                     with_attention = FLAGS.attention,
                     beam_search = FLAGS.beam_search,
                     beam_buckets = _beam_buckets
                     )

    ckpt = tf.train.get_checkpoint_state(FLAGS.saved_model_dir)
    # if FLAGS.recommend or (not FLAGS.fromScratch) and ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):

    if FLAGS.mode == "DUMP_LSTM" or FLAGS.mode == "BEAM_DECODE" or FLAGS.mode == 'FORCE_DECODE' or (not FLAGS.fromScratch) and ckpt:

        mylog("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        mylog("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

def train():

    # Read Data
    mylog_section("READ DATA")

    from_train = None
    to_train = None
    from_dev = None
    to_dev = None

    from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
        FLAGS.data_cache_dir,
        FLAGS.train_path_from,
        FLAGS.train_path_to,
        FLAGS.dev_path_from,
        FLAGS.dev_path_to,
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size)


    train_data_bucket = read_data(from_train,to_train)
    dev_data_bucket = read_data(from_dev,to_dev)
    _,_,real_vocab_size_from,real_vocab_size_to = data_utils.get_vocab_info(FLAGS.data_cache_dir)
    
    FLAGS._buckets = _buckets
    FLAGS.real_vocab_size_from = real_vocab_size_from
    FLAGS.real_vocab_size_to = real_vocab_size_to

    #train_n_tokens = total training target size
    train_n_tokens = np.sum([np.sum([len(items[1]) for items in x]) for x in train_data_bucket])
    train_bucket_sizes = [len(train_data_bucket[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]
    dev_bucket_sizes = [len(dev_data_bucket[b]) for b in xrange(len(_buckets))]
    dev_total_size = int(sum(dev_bucket_sizes))

    mylog_section("REPORT")
    # steps
    batch_size = FLAGS.batch_size
    n_epoch = FLAGS.n_epoch
    steps_per_epoch = int(train_total_size / batch_size)
    steps_per_dev = int(dev_total_size / batch_size)
    steps_per_checkpoint = int(steps_per_epoch / 2)
    total_steps = steps_per_epoch * n_epoch

    # reports
    mylog("from_vocab_size: {}".format(FLAGS.from_vocab_size))
    mylog("to_vocab_size: {}".format(FLAGS.to_vocab_size))
    mylog("_buckets: {}".format(FLAGS._buckets))
    mylog("Train:")
    mylog("total: {}".format(train_total_size))
    mylog("bucket sizes: {}".format(train_bucket_sizes))
    mylog("Dev:")
    mylog("total: {}".format(dev_total_size))
    mylog("bucket sizes: {}".format(dev_bucket_sizes))
    mylog("Steps_per_epoch: {}".format(steps_per_epoch))
    mylog("Total_steps:{}".format(total_steps))
    mylog("Steps_per_checkpoint: {}".format(steps_per_checkpoint))


    mylog_section("IN TENSORFLOW")
    
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)
    config.gpu_options.allow_growth = FLAGS.allow_growth

    with tf.Session(config=config) as sess:
        
        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog_section("MODEL/SUMMARY/WRITER")

        mylog("Creating Model.. (this can take a few minutes)")
        model = create_model(sess, run_options, run_metadata)

        if FLAGS.with_summary:
            mylog("Creating ModelSummary")
            modelSummary = ModelSummary()

            mylog("Creating tf.summary.FileWriter")
            summaryWriter = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir , "train.summary"), sess.graph)

        mylog_section("All Variables")
        show_all_variables()

        # Data Iterators
        mylog_section("Data Iterators")

        dite = DataIterator(model, train_data_bucket, len(train_buckets_scale), batch_size, train_buckets_scale)
        
        iteType = 0
        if iteType == 0:
            mylog("Itetype: withRandom")
            ite = dite.next_random()
        elif iteType == 1:
            mylog("Itetype: withSequence")
            ite = dite.next_sequence()
        
        # statistics during training
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        low_ppx = float("inf")
        low_ppx_step = 0
        steps_per_report = 30
        n_targets_report = 0
        report_time = 0
        n_valid_sents = 0
        n_valid_words = 0
        patience = FLAGS.patience
        
        mylog_section("TRAIN")

        
        while current_step < total_steps:
            
            # start
            start_time = time.time()
            
            # data and train
            source_inputs, target_inputs, target_outputs, target_weights, bucket_id = ite.next()

            L = model.step(sess, source_inputs, target_inputs, target_outputs, target_weights, bucket_id)
            
            # loss and time
            step_time += (time.time() - start_time) / steps_per_checkpoint

            loss += L
            current_step += 1
            n_valid_sents += np.sum(np.sign(target_weights[0]))
            n_valid_words += np.sum(target_weights)

            # for report
            report_time += (time.time() - start_time)
            n_targets_report += np.sum(target_weights)

            if current_step % steps_per_report == 0:
                sect_name = "STEP {}".format(current_step)
                msg = "StepTime: {:.2f} sec Speed: {:.2f} targets/s Total_targets: {}".format(report_time/steps_per_report, n_targets_report*1.0 / report_time, train_n_tokens)
                mylog_line(sect_name,msg)

                report_time = 0
                n_targets_report = 0
                

                # Create the Timeline object, and write it to a json
                if FLAGS.profile:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)
                    exit()


            
            if current_step % steps_per_checkpoint == 0:

                i_checkpoint = int(current_step / steps_per_checkpoint)
                
                # train_ppx
                loss = loss / n_valid_words
                train_ppx = math.exp(float(loss)) if loss < 300 else float("inf")
                learning_rate = model.learning_rate.eval()
                
                                
                # dev_ppx
                dev_loss, dev_ppx = evaluate(sess, model, dev_data_bucket)

                # report
                sect_name = "CHECKPOINT {} STEP {}".format(i_checkpoint, current_step)
                msg = "Learning_rate: {:.4f} Dev_ppx: {:.2f} Train_ppx: {:.2f}".format(learning_rate, dev_ppx, train_ppx)
                mylog_line(sect_name, msg)

                if FLAGS.with_summary:
                    # save summary
                    _summaries = modelSummary.step_record(sess, train_ppx, dev_ppx)
                    for _summary in _summaries:
                        summaryWriter.add_summary(_summary, i_checkpoint)
                
                # save model per checkpoint
                if FLAGS.saveCheckpoint:
                    checkpoint_path = os.path.join(FLAGS.saved_model_dir, "model")
                    s = time.time()
                    model.saver.save(sess, checkpoint_path, global_step=i_checkpoint, write_meta_graph = False)
                    msg = "Model saved using {:.2f} sec at {}".format(time.time()-s, checkpoint_path)
                    mylog_line(sect_name, msg)
                    
                # save best model
                if dev_ppx < low_ppx:
                    patience = FLAGS.patience
                    low_ppx = dev_ppx
                    low_ppx_step = current_step
                    checkpoint_path = os.path.join(FLAGS.saved_model_dir, "best")
                    s = time.time()
                    model.best_saver.save(sess, checkpoint_path, global_step=0, write_meta_graph = False)
                    msg = "Model saved using {:.2f} sec at {}".format(time.time()-s, checkpoint_path)
                    mylog_line(sect_name, msg)
                else:
                    patience -= 1

                if patience <= 0:
                    mylog("Training finished. Running out of patience.")
                    break

                # Save checkpoint and zero timer and loss.
                step_time, loss, n_valid_sents, n_valid_words = 0.0, 0.0, 0, 0
                


def evaluate(sess, model, data_set):
    # Run evals on development set and print their perplexity/loss.
    dropoutRateRaw = FLAGS.keep_prob
    sess.run(model.dropout10_op)

    start_id = 0
    loss = 0.0
    n_steps = 0
    n_valids = 0
    batch_size = FLAGS.batch_size
    
    dite = DataIterator(model, data_set, len(FLAGS._buckets), batch_size, None)
    ite = dite.next_sequence(stop = True)

    for sources, inputs, outputs, weights, bucket_id in ite:
        L = model.step(sess, sources, inputs, outputs, weights, bucket_id, forward_only = True)
        loss += L
        n_steps += 1
        n_valids += np.sum(weights)

    loss = loss/(n_valids)
    ppx = math.exp(loss) if loss < 300 else float("inf")

    sess.run(model.dropoutAssign_op)

    return loss, ppx


def force_decode():
    # force_decode it: generate a file which contains every score and the final score; 
    mylog_section("READ DATA")

    test_data_bucket, _buckets, test_data_order = read_test(FLAGS.data_cache_dir, FLAGS.test_path, get_vocab_path(FLAGS.data_cache_dir), FLAGS.L, FLAGS.n_bucket)
    vocab_path = get_vocab_path(FLAGS.data_cache_dir)
    real_vocab_size = get_real_vocab_size(vocab_path)

    FLAGS._buckets = _buckets
    FLAGS.real_vocab_size = real_vocab_size

    test_bucket_sizes = [len(test_data_bucket[b]) for b in xrange(len(_buckets))]
    test_total_size = int(sum(test_bucket_sizes))

    # reports
    mylog_section("REPORT")
    mylog("real_vocab_size: {}".format(FLAGS.real_vocab_size))
    mylog("_buckets:{}".format(FLAGS._buckets))
    mylog("FORCE_DECODE:")
    mylog("total: {}".format(test_total_size))
    mylog("bucket_sizes: {}".format(test_bucket_sizes))
    
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)
    config.gpu_options.allow_growth = FLAGS.allow_growth

    mylog_section("IN TENSORFLOW")
    with tf.Session(config=config) as sess:

        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog("Creating Model")
        model = create_model(sess, run_options, run_metadata)
                
        mylog_section("All Variables")
        show_all_variables()
 
        sess.run(model.dropoutRate.assign(1.0))

        start_id = 0
        n_steps = 0
        batch_size = FLAGS.batch_size

        mylog_section("Data Iterators")
        dite = DataIterator(model, test_data_bucket, len(_buckets), batch_size, None, data_order = test_data_order)
        ite = dite.next_original()
            
        fdump = open(FLAGS.score_file,'w')

        i_sent = 0

        mylog_section("FORCE_DECODING")

        for inputs, outputs, weights, bucket_id in ite:
            # inputs: [[_GO],[1],[2],[3],[_EOS],[pad_id],[pad_id]]
            # positions: [4]

            mylog("--- decoding {}/{} sent ---".format(i_sent, test_total_size))
            i_sent += 1
            #print(inputs)
            #print(outputs)
            #print(weights)
            #print(bucket_id)

            L = model.step(sess, inputs, outputs, weights, bucket_id, forward_only = True, dump_lstm = False)
            
            mylog("LOSS: {}".format(L))

            fdump.write("{}\n".format(L))
        
            # do the following convert:
            # inputs: [[pad_id],[1],[2],[pad_id],[pad_id],[pad_id]]
            # positions:[2]

        fdump.close()
            



def beam_decode():

    mylog("Reading Data...")

    from_test = None

    from_vocab_path, to_vocab_path, real_vocab_size_from, real_vocab_size_to = data_utils.get_vocab_info(FLAGS.data_cache_dir)
    
    FLAGS._buckets = _buckets
    FLAGS._beam_buckets = _beam_buckets
    FLAGS.real_vocab_size_from = real_vocab_size_from
    FLAGS.real_vocab_size_to = real_vocab_size_to
    
    from_test = data_utils.prepare_test_data(
        FLAGS.data_cache_dir,
        FLAGS.test_path_from,
        from_vocab_path)

    test_data_bucket, test_data_order = read_data_test(from_test)

    test_bucket_sizes = [len(test_data_bucket[b]) for b in xrange(len(_beam_buckets))]
    test_total_size = int(sum(test_bucket_sizes))

    # reports
    mylog("from_vocab_size: {}".format(FLAGS.from_vocab_size))
    mylog("to_vocab_size: {}".format(FLAGS.to_vocab_size))
    mylog("_beam_buckets: {}".format(FLAGS._beam_buckets))
    mylog("BEAM_DECODE:")
    mylog("total: {}".format(test_total_size))
    mylog("buckets: {}".format(test_bucket_sizes))
    


    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)
    config.gpu_options.allow_growth = FLAGS.allow_growth

    with tf.Session(config=config) as sess:

        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog("Creating Model")
        model = create_model(sess, run_options, run_metadata)
        show_all_variables()

        sess.run(model.dropoutRate.assign(1.0))

        start_id = 0
        n_steps = 0
        batch_size = FLAGS.batch_size
    
        dite = DataIterator(model, test_data_bucket, len(_beam_buckets), batch_size, None, data_order = test_data_order)
        ite = dite.next_original()

            
        i_sent = 0

        targets = []

        for source_inputs, bucket_id, length in ite:

            print("--- decoding {}/{} sent ---".format(i_sent, test_total_size))
            i_sent += 1
            

            results = [] # (sentence,score)
            scores = [0.0] * FLAGS.beam_size
            sentences = [[] for x in xrange(FLAGS.beam_size)]
            beam_parent = range(FLAGS.beam_size)

            target_inputs = [data_utils.GO_ID] * FLAGS.beam_size
            min_target_length = int(length * FLAGS.min_ratio) + 1
            max_target_length = int(length * FLAGS.max_ratio) + 1 # include EOS
            for i in xrange(max_target_length):
                if i == 0:
                    top_value, top_index, eos_value = model.beam_step(sess, bucket_id, index=i, sources = source_inputs, target_inputs = target_inputs)
                else:

                    top_value, top_index, eos_value = model.beam_step(sess, bucket_id, index=i,  target_inputs = target_inputs, beam_parent = beam_parent)

                # top_value = [array[batch_size, batch_size]]
                # top_index = [array[batch_size, batch_size]]
                # eos_value = [array[batch_size, 1] ]
                                        
                # expand
                global_queue = []

                if i == 0:
                    nrow = 1
                else:
                    nrow = FLAGS.beam_size

                if i == max_target_length - 1: # last_step
                    for row in xrange(nrow):

                        score = scores[row] + np.log(eos_value[0][row,0])
                        word_index = data_utils.EOS_ID
                        beam_index = row
                        global_queue.append((score, beam_index, word_index))                         

                else:
                    for row in xrange(nrow):
                        for col in xrange(top_index[0].shape[1]):
                            score = scores[row] + np.log(top_value[0][row,col])
                            word_index = top_index[0][row,col]
                            beam_index = row

                            global_queue.append((score, beam_index, word_index))                         

                global_queue = sorted(global_queue, key = lambda x : -x[0])


                if FLAGS.print_beam:
                    print("--------- Step {} --------".format(i))

                target_inputs = []
                beam_parent = []
                scores = []
                temp_sentences = []

                for j, (score, beam_index, word_index) in enumerate(global_queue):
                    if word_index == data_utils.EOS_ID:
                        if len(sentences[beam_index])+1 < min_target_length:
                            continue

                        results.append((sentences[beam_index] + [word_index], score))
                        if FLAGS.print_beam:
                            print("*Beam:{} Father:{} word:{} score:{}".format(j,beam_index,word_index,score))
                        continue
                    
                    if FLAGS.print_beam:
                        print("Beam:{} Father:{} word:{} score:{}".format(j,beam_index,word_index,score))
                    beam_parent.append(beam_index)

                    
                    target_inputs.append(word_index)
                    scores.append(score)
                    temp_sentences.append(sentences[beam_index] + [word_index])
                    
                    if len(scores) >= FLAGS.beam_size:
                        break
                   
                # can not fill beam_size, just repeat the last one
                while len(scores) < FLAGS.beam_size and i < max_target_length - 1:
                    beam_parent.append(beam_parent[-1])
                    target_inputs.append(target_inputs[-1])
                    scores.append(scores[-1])
                    temp_sentences.append(temp_sentences[-1])
                
                sentences = temp_sentences
                    
                # print the 1 best 
            results = sorted(results, key = lambda x: -x[1])
            
            targets.append(results[0][0])

        data_utils.ids_to_tokens(targets, to_vocab_path, FLAGS.decode_output)
                


           
def dump_lstm():
    # dump the hidden states to some where
    mylog_section("READ DATA")
    test_data_bucket, _buckets, test_data_order = read_test(FLAGS.data_cache_dir, FLAGS.test_path, get_vocab_path(FLAGS.data_cache_dir), FLAGS.L, FLAGS.n_bucket)
    vocab_path = get_vocab_path(FLAGS.data_cache_dir)
    real_vocab_size = get_real_vocab_size(vocab_path)

    FLAGS._buckets = _buckets
    FLAGS.real_vocab_size = real_vocab_size

    test_bucket_sizes = [len(test_data_bucket[b]) for b in xrange(len(_buckets))]
    test_total_size = int(sum(test_bucket_sizes))

    # reports
    mylog_section("REPORT")

    mylog("real_vocab_size: {}".format(FLAGS.real_vocab_size))
    mylog("_buckets:{}".format(FLAGS._buckets))
    mylog("DUMP_LSTM:")
    mylog("total: {}".format(test_total_size))
    mylog("buckets: {}".format(test_bucket_sizes))
    
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)
    config.gpu_options.allow_growth = FLAGS.allow_growth
    with tf.Session(config=config) as sess:

        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog_section("MODEL")

        mylog("Creating Model")
        model = create_model(sess, run_options, run_metadata)
        
        mylog("Init tensors to dump")
        model.init_dump_states()

        #dump_graph('graph.txt')
        mylog_section("All Variables")
        show_all_variables()
 
        sess.run(model.dropoutRate.assign(1.0))

        start_id = 0
        n_steps = 0
        batch_size = FLAGS.batch_size

        mylog_section("Data Iterators")

        dite = DataIterator(model, test_data_bucket, len(_buckets), batch_size, None, data_order = test_data_order)
        ite = dite.next_original()
            
        fdump = open(FLAGS.dump_file,'wb')

        mylog_section("DUMP_LSTM")

        i_sent = 0
        for inputs, outputs, weights, bucket_id in ite:
            # inputs: [[_GO],[1],[2],[3],[_EOS],[pad_id],[pad_id]]
            # positions: [4]

            mylog("--- decoding {}/{} sent ---".format(i_sent, test_total_size))
            i_sent += 1
            #print(inputs)
            #print(outputs)
            #print(weights)
            #print(bucket_id)

            L, states = model.step(sess, inputs, outputs, weights, bucket_id, forward_only = True, dump_lstm = True)
            
            mylog("LOSS: {}".format(L))
            
            sw = StateWrapper()
            sw.create(inputs,outputs,weights,states)
            sw.save_to_stream(fdump)
            
            # do the following convert:
            # inputs: [[pad_id],[1],[2],[pad_id],[pad_id],[pad_id]]
            # positions:[2]

        fdump.close()
        
    



def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def parsing_flags():
    # saved_model

    FLAGS.data_cache_dir = os.path.join(FLAGS.model_dir, "data_cache")
    FLAGS.saved_model_dir = os.path.join(FLAGS.model_dir, "saved_model")
    FLAGS.summary_dir = FLAGS.saved_model_dir

    mkdir(FLAGS.model_dir)
    mkdir(FLAGS.data_cache_dir)
    mkdir(FLAGS.saved_model_dir)
    mkdir(FLAGS.summary_dir)

    # for logs
    log_path = os.path.join(FLAGS.model_dir,"log.{}.txt".format(FLAGS.mode))
    filemode = 'w' if FLAGS.fromScratch else "a"
    logging.basicConfig(filename=log_path,level=logging.DEBUG, filemode = filemode)
    
    FLAGS.beam_search = False

    log_flags()

    
 
def main(_):
    
    parsing_flags()
    
    if FLAGS.mode == "TRAIN":
        train()


    # not ready yet
    if FLAGS.mode == 'FORCE_DECODE':
        mylog("\nWARNING: \n 1. The output file and original file may not align one to one, because we remove the lines whose lenght exceeds the maximum length set by -L \n 2. The score is -sum(log(p)) with base e and includes EOS. \n")
        
        FLAGS.batch_size = 1
        FLAGS.score_file = os.path.join(FLAGS.model_dir,FLAGS.force_decode_output)
        #FLAGS.n_bucket = 1
        force_decode()

    # not ready yet
    if FLAGS.mode == 'DUMP_LSTM':
        mylog("\nWARNING: The output file and original file may not align one to one, because we remove the lines whose lenght exceeds the maximum length set by -L \n")
            
        FLAGS.batch_size = 1
        FLAGS.dump_file = os.path.join(FLAGS.model_dir,FLAGS.dump_lstm_output)
        #FLAGS.n_bucket = 1
        dump_lstm()

    if FLAGS.mode == "BEAM_DECODE":
        FLAGS.batch_size = FLAGS.beam_size
        FLAGS.beam_search = True
        beam_decode()
    
    logging.shutdown()
    
if __name__ == "__main__":
    tf.app.run()
