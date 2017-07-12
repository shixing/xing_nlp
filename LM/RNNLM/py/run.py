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
from data_util import read_train_dev, read_test, get_real_vocab_size, get_vocab_path
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
tf.app.flags.DEFINE_string("train_path", "./train", "the absolute path of raw train file.")
tf.app.flags.DEFINE_string("dev_path", "./dev", "the absolute path of raw dev file.")
tf.app.flags.DEFINE_string("test_path", "./test", "the absolute path of raw test file.")
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
tf.app.flags.DEFINE_integer("vocab_size", 10000, "vocabulary size.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("n_epoch", 500,
                            "Maximum number of epochs in training.")
tf.app.flags.DEFINE_integer("L", 30,"max length")
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
tf.app.flags.DEFINE_integer("beam_step", 3,"the beam step")
tf.app.flags.DEFINE_integer("topk", 3,"topk")
tf.app.flags.DEFINE_boolean("print_beam", False, "to print beam info")
tf.app.flags.DEFINE_boolean("no_repeat", False, "no repeat")

# GPU configuration
tf.app.flags.DEFINE_boolean("allow_growth", False, "allow growth")



FLAGS = tf.app.flags.FLAGS



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
                     FLAGS.real_vocab_size,
                     FLAGS.num_layers,
                     FLAGS.max_gradient_norm,
                     FLAGS.batch_size,
                     FLAGS.learning_rate,
                     FLAGS.learning_rate_decay_factor,
                     withAdagrad = FLAGS.withAdagrad,
                     dropoutRate = FLAGS.keep_prob,
                     dtype = dtype,
                     devices = devices,
                     topk_n = FLAGS.topk,
                     run_options = run_options,
                     run_metadata = run_metadata
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
    train_data_bucket, dev_data_bucket, _buckets, vocab_path = read_train_dev(FLAGS.data_cache_dir, FLAGS.train_path, FLAGS.dev_path, FLAGS.vocab_size, FLAGS.L, FLAGS.n_bucket)
    real_vocab_size = get_real_vocab_size(vocab_path)

    FLAGS._buckets = _buckets
    FLAGS.real_vocab_size = real_vocab_size

    train_n_tokens = np.sum([np.sum([len(items) for items in x]) for x in train_data_bucket])
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
    mylog("real_vocab_size: {}".format(FLAGS.real_vocab_size))
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
            inputs, outputs, weights, bucket_id = ite.next()

            L = model.step(sess, inputs, outputs, weights, bucket_id)
            
            # loss and time
            step_time += (time.time() - start_time) / steps_per_checkpoint

            loss += L
            current_step += 1
            n_valid_sents += np.sum(np.sign(weights[0]))
            n_valid_words += np.sum(weights)

            # for report
            report_time += (time.time() - start_time)
            n_targets_report += np.sum(weights)

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

    for inputs, outputs, weights, bucket_id in ite:
        L = model.step(sess, inputs, outputs, weights, bucket_id, forward_only = True)
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
    # not yet tested: 
    # known issues:
    #   should use next_original
    mylog("Reading Data...")
    test_data_bucket, _buckets, test_data_order = read_test(FLAGS.data_cache_dir, FLAGS.test_path, get_vocab_path(FLAGS.data_cache_dir), FLAGS.L, FLAGS.n_bucket)
    vocab_path = get_vocab_path(FLAGS.data_cache_dir)
    real_vocab_size = get_real_vocab_size(vocab_path)

    FLAGS._buckets = _buckets
    FLAGS.real_vocab_size = real_vocab_size

    test_bucket_sizes = [len(test_data_bucket[b]) for b in xrange(len(_buckets))]
    test_total_size = int(sum(test_bucket_sizes))

    # reports
    mylog("real_vocab_size: {}".format(FLAGS.real_vocab_size))
    mylog("_buckets:{}".format(FLAGS._buckets))
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
        mylog("before init_beam_decoder()")
        show_all_variables()
        model.init_beam_decoder(beam_size = FLAGS.beam_size, max_steps = FLAGS.beam_step)
        model.init_beam_variables(sess)
        mylog("after init_beam_decoder()")
        show_all_variables()

        sess.run(model.dropoutRate.assign(1.0))

        start_id = 0
        n_steps = 0
        batch_size = FLAGS.batch_size
    
        dite = DataIterator(model, test_data_bucket, len(_buckets), batch_size, None)
        ite = dite.next_sequence(stop = True, test = True)

            
        i_sent = 0
        for inputs, positions, valids, bucket_id in ite:
            # user : [0]
            # inputs: [[_GO],[1],[2],[3],[_EOS],[pad_id],[pad_id]]
            # positions: [4]



            print("--- decoding {}/{} sent ---".format(i_sent, n_total_user))
            i_sent += 1
            
            # do the following convert:
            # inputs: [[pad_id],[1],[2],[pad_id],[pad_id],[pad_id]]
            # positions:[2]
            PAD_ID = 0
            last_history = inputs[positions[0]]
            inputs_beam = [last_history * FLAGS.beam_size]
            inputs[positions[0]] = list([PAD_ID] * FLAGS.beam_size)
            inputs[positions[0]-1] = list([PAD_ID] * FLAGS.beam_size)
            positions[0] = positions[0] - 2 if positions[0] >= 2 else 0            
            scores = [0.0] * FLAGS.beam_size
            sentences = [[] for x in xrange(FLAGS.beam_size)]
            beam_parent = range(FLAGS.beam_size)

            for i in xrange(FLAGS.beam_step):
                if i == 0:
                    top_value, top_index = model.beam_step(sess, index=i, word_inputs_history = inputs, sequence_length = positions,  word_inputs_beam = inputs_beam)
                else:
                    top_value, top_index = model.beam_step(sess, index=i,  word_inputs_beam = inputs_beam, beam_parent = beam_parent)
                    
                # expand
                global_queue = []

                if i == 0:
                    nrow = 1
                else:
                    nrow = top_index[0].shape[0]

                for row in xrange(nrow):
                    for col in xrange(top_index[0].shape[1]):
                        score = scores[row] + np.log(top_value[0][row,col])
                        word_index = top_index[0][row,col]
                        beam_index = row
                        
                        if FLAGS.no_repeat:
                            if not word_index in sentences[beam_index]:
                                global_queue.append((score, beam_index, word_index))
                        else:
                            global_queue.append((score, beam_index, word_index))                         

                global_queue = sorted(global_queue, key = lambda x : -x[0])

                inputs_beam = []
                beam_parent = []
                scores = []
                temp_sentences = []

                if FLAGS.print_beam:
                    print("--------- Step {} --------".format(i))

                for j, (score, beam_index, word_index) in enumerate(global_queue[:FLAGS.beam_size]):
                    if FLAGS.print_beam:
                        print("Beam:{} Father:{} word:{} score:{}".format(j,beam_index,word_index,score))
                    beam_parent.append(beam_index)
                    inputs_beam.append(word_index)
                    scores.append(score)
                    temp_sentences.append(sentences[beam_index] + [word_index])
                    
                inputs_beam = [inputs_beam]
                sentences = temp_sentences

            if FLAGS.print_beam:
                print(sentences)


           
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
    # add data_cache and saved_model
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
    
    
    log_flags()

    
 
def main(_):
    
    parsing_flags()
    
    if FLAGS.mode == "TRAIN":
        train()

    if FLAGS.mode == 'FORCE_DECODE':
        mylog("\nWARNING: \n 1. The output file and original file may not align one to one, because we remove the lines whose lenght exceeds the maximum length set by -L \n 2. The score is -sum(log(p)) with base e and includes EOS. \n")
        
        FLAGS.batch_size = 1
        FLAGS.score_file = os.path.join(FLAGS.model_dir,FLAGS.force_decode_output)
        #FLAGS.n_bucket = 1
        force_decode()


    if FLAGS.mode == 'DUMP_LSTM':
        mylog("\nWARNING: The output file and original file may not align one to one, because we remove the lines whose lenght exceeds the maximum length set by -L \n")
            
        FLAGS.batch_size = 1
        FLAGS.dump_file = os.path.join(FLAGS.model_dir,FLAGS.dump_lstm_output)
        #FLAGS.n_bucket = 1
        dump_lstm()

    if FLAGS.mode == "BEAM_DECODE":
        FLAGS.batch_size = 1
        FLAGS.n_bucket = 1
        beam_decode()
    
    logging.shutdown()
    
if __name__ == "__main__":
    tf.app.run()
