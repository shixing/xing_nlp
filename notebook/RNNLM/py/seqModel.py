import random

import numpy as np
# from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope

import data_iterator
# import env

class SeqModel(object):
    
    def __init__(self,
                 buckets,
                 size,
                 vocab_size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 withAdagrad = True,
                 forward_only=False,
                 dropoutRate = 1.0,
                 devices = "",
                 run_options = None,
                 run_metadata = None,
                 topk_n = 30,
                 dtype=tf.float32,
                 ):
        """Create the model.
        
        Args:
        buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
        size: number of units in each layer of the model.
        num_layers: number of layers in the model.
        max_gradient_norm: gradients will be clipped to maximally this norm.
        batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.

        learning_rate: learning rate to start with.
        learning_rate_decay_factor: decay learning rate by this much when needed.

        forward_only: if set, we do not construct the backward pass in the model.
        dtype: the data type to use to store internal variables.
        """
        self.buckets = buckets
        self.PAD_ID = 0
        self.batch_size = batch_size
        self.devices = devices
        self.run_options = run_options
        self.run_metadata = run_metadata
        self.topk_n = topk_n
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # some parameters
        with tf.device(devices[0]):
            self.dropoutRate = tf.Variable(
                float(dropoutRate), trainable=False, dtype=dtype)        
            self.dropoutAssign_op = self.dropoutRate.assign(dropoutRate)
            self.dropout10_op = self.dropoutRate.assign(1.0)
            self.learning_rate = tf.Variable(
                float(learning_rate), trainable=False, dtype=dtype)
            self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False)


        # Input Layer
        with tf.device(devices[0]):
            self.inputs = []
            self.inputs_embed = []
            
            self.input_embedding = tf.get_variable("input_embeddiing",[vocab_size, size], dtype = dtype)
            # 建立最大长度的inputs
            for i in range(buckets[-1]):
                input_plhd = tf.placeholder(tf.int32, shape = [self.batch_size], name = "input{}".format(i))
                input_embed = tf.nn.embedding_lookup(self.input_embedding, input_plhd)
                self.inputs.append(input_plhd)
                self.inputs_embed.append(input_embed)


        def lstm_cell():
            cell = tf.contrib.rnn.LSTMCell(size, state_is_tuple=True)
            # 建立输入的 dropout
            cell = tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob = self.dropoutRate)
            return cell

        # LSTM
        # with tf.device 可以让每个计算都绑定到不同的 cpu 上
        with tf.device(devices[1]):
            if num_layers == 1:
                single_cell = lstm_cell()
            else:
                single_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
            single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob = self.dropoutRate)
        # 建立单个 rnn，多层，带 dropout
        self.single_cell = single_cell
        
        
        # Output Layer
        with tf.device(devices[2]):
            self.targets = []
            self.target_weights = []
            
            self.output_embedding = tf.get_variable("output_embeddiing",[vocab_size, size], dtype = dtype)
            self.output_bias = tf.get_variable("output_bias",[vocab_size], dtype = dtype)

            # target: 1  2  3  4 
            # inputs: go 1  2  3
            # weights:1  1  1  1

            for i in range(buckets[-1]):
                self.targets.append(tf.placeholder(tf.int32, 
                    shape=[self.batch_size], name = "target{}".format(i)))
                self.target_weights.append(tf.placeholder(dtype, 
                    shape = [self.batch_size], name="target_weight{}".format(i)))
            

        
        
        # Model with buckets
        # 对于多 buckets 我们需要对于每个 buckets 都需要计算 loss 和 update 操作
        self.model_with_buckets(self.inputs_embed, self.targets, self.target_weights, self.buckets, single_cell, dtype, devices = devices)


        # train
        with tf.device(devices[0]):
            params = tf.trainable_variables()
            # 不仅前向计算 forward, backward, update
            if not forward_only:
                self.gradient_norms = []
                self.updates = []
                if withAdagrad:
                    opt = tf.train.AdagradOptimizer(self.learning_rate)
                else:
                    opt = tf.train.GradientDescentOptimizer(self.learning_rate)

                for b in range(len(buckets)):
                    gradients = tf.gradients(self.losses[b], params, colocate_gradients_with_ops=True)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())
        self.best_saver = tf.train.Saver(tf.global_variables())


    def get_hidden_states(self,bucket_id, max_length, n_layers):
        states = []
        def get_name(istep,ilayer,name):
            d = {"fg":"Sigmoid",'ig':"Sigmoid_1",'og':"Sigmoid_2",'i':"Tanh",'h':"mul_2",'c':"add_1"}
            step_str = ''
            if istep > 0:
                step_str = "_{}".format(istep)
            bucket_str = ""
            if bucket_id > 0:
                bucket_str = "_{}".format(bucket_id)
            return "model_with_buckets/rnn{}/multi_rnn_cell{}/cell_{}/lstm_cell/{}:0".format(bucket_str, step_str, ilayer,d[name])

        names = ['fg','ig','og','i','h','c']
        graph = tf.get_default_graph()
        for i in range(max_length):
            state_step = []
            for j in range(n_layers):
                state_layer = {}
                for name in names:
                    tensor = graph.get_tensor_by_name(get_name(i,j,name))
                    state_layer[name] = tensor
                state_step.append(state_layer)
            states.append(state_step)
        return states 


    def init_dump_states(self):
        self.states_to_dump = []
        for i, l in enumerate(self.buckets):
            states = self.get_hidden_states(i,l,self.num_layers)
            self.states_to_dump.append(states)


    def init_beam_decoder(self,beam_size=10, max_steps = 30):

        # a non bucket design
        #  
        # how to feed in: 
        # inputs = [GO, 1, 2, 3], sequene_length = [4-1]
        
        # NOTE: device allocation is not ready yet. 
        


        self.beam_size = beam_size

        init_state = self.single_cell.zero_state(1, self.dtype)
        if self.num_layers == 1:
            init_state = [init_state]

        self.before_state = []
        self.after_state = []
        print(init_state)
        shape = [self.beam_size, init_state[0].c.get_shape()[1]]

        with tf.device(self.devices[0]):

            with tf.variable_scope("beam_search"):

                # two variable: before_state, after_state
                for i, state_tuple in enumerate(init_state):
                    cb = tf.get_variable("before_c_{}".format(i), shape, initializer=tf.constant_initializer(0.0), trainable = False) 
                    hb = tf.get_variable("before_h_{}".format(i), shape, initializer=tf.constant_initializer(0.0), trainable = False) 
                    sb = tf.nn.rnn_cell.LSTMStateTuple(cb,hb)
                    ca = tf.get_variable("after_c_{}".format(i), shape, initializer=tf.constant_initializer(0.0), trainable = False) 
                    ha = tf.get_variable("after_h_{}".format(i), shape, initializer=tf.constant_initializer(0.0), trainable = False) 
                    sa = tf.nn.rnn_cell.LSTMStateTuple(ca,ha)
                    self.before_state.append(sb)
                    self.after_state.append(sa)                

                # a new place holder for sequence_length 
                self.sequence_length = tf.placeholder(tf.int32, shape=[1], name = "sequence_length")
                
            # the final_state after processing the start state 
            with tf.variable_scope("",reuse=True):
                if len(init_state) == 1:
                    _, beam_final_state = rnn.rnn(self.single_cell,self.inputs_embed,initial_state = init_state[0], sequence_length = self.sequence_length)
                    self.beam_final_state = [beam_final_state]
                else:
                    _, self.beam_final_state = rnn.rnn(self.single_cell,self.inputs_embed, initial_state = init_state, sequence_length = self.sequence_length)
                
            with tf.variable_scope("beam_search"):
                # copy the final_state to before_state
                self.final2before_ops = [] # an operation sequence
                for i in range(len(self.before_state)):
                    final_c = self.beam_final_state[i].c
                    final_h = self.beam_final_state[i].h
                    final_c_expand = tf.nn.embedding_lookup(final_c,[0] * self.beam_size)
                    final_h_expand = tf.nn.embedding_lookup(final_h,[0] * self.beam_size)
                    copy_c = self.before_state[i].c.assign(final_c_expand)
                    copy_h = self.before_state[i].h.assign(final_h_expand)
                    self.final2before_ops.append(copy_c)
                    self.final2before_ops.append(copy_h)

                # operation: copy after_state to before_state according to a ma
                self.beam_parent = tf.placeholder(tf.int32, shape=[self.beam_size], name = "beam_parent")
                self.after2before_ops = [] # an operation sequence
                for i in range(len(self.before_state)):
                    after_c = self.after_state[i].c
                    after_h = self.after_state[i].h
                    after_c_expand = tf.nn.embedding_lookup(after_c,self.beam_parent)
                    after_h_expand = tf.nn.embedding_lookup(after_h,self.beam_parent)
                    copy_c = self.before_state[i].c.assign(after_c_expand)
                    copy_h = self.before_state[i].h.assign(after_h_expand)
                    self.after2before_ops.append(copy_c)
                    self.after2before_ops.append(copy_h)


            # operation: one step RNN 
            with tf.variable_scope("",reuse=True):

                # Input Layer
                self.beam_inputs = []
                self.beam_inputs_embed = []

                for i in range(1): # only one step 
                    beam_input_plhd = tf.placeholder(tf.int32, shape = [self.batch_size], name = "beam_input{}".format(i))
                    beam_input_embed = tf.nn.embedding_lookup(self.input_embedding, input_plhd)
                    self.beam_inputs.append(input_plhd)
                    self.beam_inputs_embed(input_embed)
            
                if len(self.before_state) == 1:
                    self.beam_step_hts, beam_step_state = rnn.rnn(self.single_cell,self.beam_inputs_embed,initial_state = self.before_state[0])
                    self.beam_step_state = [beam_step_state]
                else:
                    self.beam_step_hts, self.beam_step_state = rnn.rnn(self.single_cell,self.beam_inputs_embed,initial_state = self.before_state)

            with tf.variable_scope("beam_search"):
                # operate: copy beam_step_state to after_state
                self.beam2after_ops = [] # an operation sequence
                for i in range(len(self.after_state)):
                    copy_c = self.after_state[i].c.assign(self.beam_step_state[i].c)
                    copy_h = self.after_state[i].h.assign(self.beam_step_state[i].h)
                    self.beam2after_ops.append(copy_c)
                    self.beam2after_ops.append(copy_h)

            self.beam_step_logits = [tf.add(tf.matmul(ht, tf.transpose(self.output_embedding)) + self.output_bias) for ht in self.beam_step_hts]
            self.beam_top_value = []
            self.beam_top_index = []
            for _logits in self.beam_step_logits:
                value, index = tf.nn.top_k(tf.nn.softmax(_logits), self.beam_size, sorted = True)
                self.beam_top_value.append(value)
                self.beam_top_index.append(index)



    def show_before_state(self):
        for i in range(self.before_state):
            print(self.before_state[i].c.eval())
            print(self.before_state[i].h.eval())

    def show_after_state(self):
        for i in range(len(self.after_state)):
            print(self.after_state[i].c.eval())
            print(self.after_state[i].h.eval())

    def beam_step(self, session, index = 0, word_inputs_history=None,sequence_length = None, word_inputs_beam = None, beam_parent = None, bucket_id = 0):

        if index == 0:            
            # go through the history by LSTM 
            input_feed = {}         
            for i in range(len(word_inputs)):
                input_feed[self.inputs[i].name] = self.word_inputs_history[i]

            input_feed[self.sequence_length.name] = sequence_length
            
            output_feed = []
            output_feed += self.final2before_ops
            _ = session.run(output_feed, input_feed)
            
        else:
            # copy the after_state to before states
            input_feed = {}
            input_feed[self.beam_parent.name] = beam_parent
            output_feed = []
            output_feed += self.after2before_ops
            _ = session.run(output_feed, input_feed)
        # Run one step of RNN

        input_feed = {}

        input_feed[self.beam_inputs[0].name] = self.word_inputs_beam[0]

        output_feed = {}
        output_feed['value'] = self.beam_top_value
        output_feed['index'] = self.beam_top_index
        output_feed['ops'] = self.beam2after_ops

        outputs = session.run(output_feed,input_feed)
        
        return outputs['value'], outputs['index']



    def step(self,session, inputs, targets, target_weights, 
        bucket_id, forward_only = False, dump_lstm = False):

        length = self.buckets[bucket_id]

        input_feed = {}
        for l in range(length):
            input_feed[self.inputs[l].name] = inputs[l]
            input_feed[self.targets[l].name] = targets[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # output_feed
        if forward_only:
            output_feed = [self.losses[bucket_id]]
            if dump_lstm:
                output_feed.append(self.states_to_dump[bucket_id])

        else:
            output_feed = [self.losses[bucket_id]]
            output_feed += [self.updates[bucket_id], self.gradient_norms[bucket_id]]

        outputs = session.run(output_feed, input_feed, options = self.run_options, run_metadata = self.run_metadata)

        if forward_only and dump_lstm:
            return outputs
        else:
            return outputs[0] # only return losses
    

    def get_batch(self, data_set, bucket_id, start_id = None):
        length = self.buckets[bucket_id]

        input_ids,output_ids, weights = [], [], []

        for i in range(self.batch_size):
            if start_id == None:
                word_seq = random.choice(data_set[bucket_id])
            else:
                if start_id + i < len(data_set[bucket_id]):
                    word_seq = data_set[bucket_id][start_id + i]
                else:
                    word_seq = []                    

            
            word_input_seq = word_seq[:-1]  # without _EOS
            word_output_seq = word_seq[1:]  # without _GO

            target_weight = [1.0] * len(word_output_seq) + [0.0] * (length - len(word_output_seq))
            word_input_seq = word_input_seq + [self.PAD_ID] * (length - len(word_input_seq))
            word_output_seq = word_output_seq + [self.PAD_ID] * (length - len(word_output_seq))

            input_ids.append(word_input_seq)
            output_ids.append(word_output_seq)
            weights.append(target_weight)
            
        # Now we create batch-major vectors from the data selected above.
        def batch_major(l):
            output = []
            for i in range(len(l[0])):
                temp = []
                for j in range(self.batch_size):
                    temp.append(l[j][i])
                output.append(temp)
            return output
            
        batch_input_ids = batch_major(input_ids)
        batch_output_ids = batch_major(output_ids)
        batch_weights = batch_major(weights)
        
        finished = False
        if start_id != None and start_id + self.batch_size >= len(data_set[bucket_id]):
            finished = True


        return batch_input_ids, batch_output_ids, batch_weights, finished

        
    def get_batch_test(self, data_set, bucket_id, start_id = None):
        length = self.buckets[bucket_id]
        
        word_inputs, positions, valids = [], [], []
        
        for i in range(self.batch_size):
            if start_id == None:
                word_seq = random.choice(data_set[bucket_id])
                valid = 1
                position = len(word_seq) - 1
            else:
                if start_id + i < len(data_set[bucket_id]):
                    word_seq = data_set[bucket_id][start_id + i]
                    valid = 1
                    position = len(word_seq) - 1
                else:                    
                    word_seq = []                    
                    valid = 0
                    position = length-1
            
            pad_seq = [self.PAD_ID] * (length - len(word_seq))
            word_input_seq = word_seq + pad_seq
            valids.append(valid)
            positions.append(position)
            word_inputs.append(word_input_seq)
            
        # Now we create batch-major vectors from the data selected above.
        def batch_major(l):
            output = []
            for i in range(len(l[0])):
                temp = []
                for j in range(self.batch_size):
                    temp.append(l[j][i])
                output.append(temp)
            return output
            
        batch_word_inputs = batch_major(word_inputs)
        
        finished = False
        if start_id != None and start_id + self.batch_size >= len(data_set[bucket_id]):
            finished = True

        return batch_word_inputs, positions, valids, finished
        
    def model_with_buckets(self, inputs, targets, weights,
                           buckets, cell, dtype,
                           per_example_loss=False, name=None, devices = None):

        all_inputs = inputs + targets + weights

        losses = []
        hts = []
        logits = []
        topk_values = []
        topk_indexes = []

        # initial state
        with tf.device(devices[1]):
            init_state = cell.zero_state(self.batch_size, dtype)

        # softmax
        with tf.device(devices[2]):
            softmax_loss_function = lambda x,y: tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels= y)


        with tf.name_scope(name, "model_with_buckets", all_inputs):
            for j, bucket in enumerate(buckets):
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),reuse=True if j > 0 else None):
                    
                    # ht
                    with tf.device(devices[1]):
                        _hts, _ = tf.contrib.rnn.static_rnn(cell,inputs[:bucket],initial_state = init_state)
                        hts.append(_hts)

                    # logits / loss / topk_values + topk_indexes
                    with tf.device(devices[2]):
                        _logits = [ tf.add(tf.matmul(ht, tf.transpose(self.output_embedding)), self.output_bias) for ht in _hts]
                        logits.append(_logits)

                        if per_example_loss:
                            losses.append(sequence_loss_by_example(
                                    logits[-1], targets[:bucket], weights[:bucket],
                                    softmax_loss_function=softmax_loss_function))
                        
                        else:
                            losses.append(sequence_loss(
                                    logits[-1], targets[:bucket], weights[:bucket],
                                    softmax_loss_function=softmax_loss_function))
                        
                        topk_value, topk_index = [], []

                        for _logits in logits[-1]:
                            value, index = tf.nn.top_k(tf.nn.softmax(_logits), self.topk_n, sorted = True)
                            topk_value.append(value)
                            topk_index.append(index)
                        topk_values.append(topk_value)
                        topk_indexes.append(topk_index)

        self.losses = losses
        self.hts = hts
        self.logits = logits
        self.topk_values = topk_values
        self.topk_indexes = topk_indexes


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with tf.name_scope(name, "sequence_loss_by_example", logits + targets + weights):
  # with ops.op_scope(logits + targets + weights,name, "sequence_loss_by_example"):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logit, target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)

    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=False, average_across_batch=False,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """

  with tf.name_scope(name, "sequence_loss", logits + targets + weights):
  # with ops.op_scope(logits + targets + weights, name, "sequence_loss"):
    cost = math_ops.reduce_sum(sequence_loss_by_example(
        logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))
    if average_across_batch:
        total_size = tf.reduce_sum(tf.sign(weights[0]))
        return cost / math_ops.cast(total_size, cost.dtype)
    else:
      return cost
