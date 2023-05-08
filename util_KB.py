#-*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import codecs
import collections
import json
import math
import shutil
import sys

import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import pyhocon


def initialize_from_env(experiment, logdir=None):

    if "GPU" in os.environ:
        set_gpus(int(os.environ["GPU"]))
    else:
        set_gpus()

    print("Running experiment: {}".format(experiment))

    # 👇，这里解析的文件改成了experiments.conf
    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[experiment]
    
    if logdir is None:
        logdir = experiment
        
    config["log_dir"] = mkdirs(os.path.join(config["log_root"], "ab_omcs"))
    #config["log_dir"] = mkdirs(os.path.join(config["log_root"], name))

    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config

def initialize_from_env0():
  if "GPU" in os.environ:
    set_gpus(int(os.environ["GPU"]))
  else:
    set_gpus()

  
  name = "att"
  #name = "finalatt"
  
  print("Running experiment: {}".format(name))

  config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]
  config["log_dir"] = mkdirs(os.path.join(config["log_root"], "attMM0.8_2"))
  
  print(pyhocon.HOCONConverter.convert(config, "hocon"))
  return config
  
def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        shutil.copyfile(source + ext, target + ext)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('experiment')
    parser.add_argument('-l', '--logdir')
    parser.add_argument('--latest-checkpoint', action='store_true')
    return parser.parse_args()
    
    
def make_summary(value_dict):
    return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])


def flatten(l):
    return [item for sublist in l for item in sublist]


def set_gpus(*gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    tf.ConfigProto().gpu_options.per_process_gpu_memory_fraction = 0.9
    tf.ConfigProto().gpu_options.allow_growth = True
    print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def load_char_dict(char_vocab_path):
    vocab = [u"<unk>"]
    with codecs.open(char_vocab_path, encoding="utf-8") as f:
        vocab.extend(l.strip() for l in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})
    return char_dict


def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)


def projection(inputs, output_size, initializer=None):
    return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)


def highway(inputs, num_layers, dropout):
    for i in range(num_layers):
        with tf.variable_scope("highway_{}".format(i)):
            j, f = tf.split(projection(inputs, 2 * shape(inputs, -1)), 2, -1)
            f = tf.sigmoid(f)
            j = tf.nn.relu(j)
            if dropout is not None:
                j = tf.nn.dropout(j, dropout)
            inputs = f * j + (1 - f) * inputs
    return inputs


def shape(x, dim):
    return x.get_shape()[dim].value or tf.shape(x)[dim]
    


def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
    if len(inputs.get_shape()) > 3:
        raise ValueError("FFNN with rank {} not supported".format(
            len(inputs.get_shape())))

    if len(inputs.get_shape()) == 3:
        batch_size = shape(inputs, 0)
        seqlen = shape(inputs, 1)
        emb_size = shape(inputs, 2)
        current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
    else:
        current_inputs = inputs

    for i in range(num_hidden_layers):
        hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
        hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
        current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

        if dropout is not None:
            current_outputs = tf.nn.dropout(current_outputs, dropout)
        current_inputs = current_outputs

    output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
    output_bias = tf.get_variable("output_bias", [output_size])
    outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

    if len(inputs.get_shape()) == 3:
        outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
    return outputs


def cnn(inputs, filter_sizes, num_filters):
    num_words = shape(inputs, 0)
    num_chars = shape(inputs, 1)
    input_size = shape(inputs, 2)
    outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv_{}".format(i)):
            w = tf.get_variable("w", [filter_size, input_size, num_filters])
            b = tf.get_variable("b", [num_filters])
        # [num_words, num_chars - filter_size, num_filters]
        conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID")
        # [num_words, num_chars - filter_size, num_filters]
        h = tf.nn.relu(tf.nn.bias_add(conv, b))
        pooled = tf.reduce_max(h, 1)  # [num_words, num_filters]
        outputs.append(pooled)
    # [num_words, num_filters * len(filter_sizes)]
    return tf.concat(outputs, 1)


def batch_gather(emb, indices):
    batch_size = shape(emb, 0)
    seqlen = shape(emb, 1)
    if len(emb.get_shape()) > 2:
        emb_size = shape(emb, 2)
    else:
        emb_size = 1
    # [batch_size * seqlen, emb]
    flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])
    offset = tf.expand_dims(tf.range(batch_size) *
                            seqlen, 1)  # [batch_size, 1]
    # [batch_size, num_indices, emb]
    gathered = tf.gather(flattened_emb, indices + offset)
    if len(emb.get_shape()) == 2:
        gathered = tf.squeeze(gathered, 2)  # [batch_size, num_indices]
    return gathered


class RetrievalEvaluator(object):
    def __init__(self):
        self._num_correct = 0
        self._num_gold = 0
        self._num_predicted = 0

    def update(self, gold_set, predicted_set):
        self._num_correct += len(gold_set & predicted_set)
        self._num_gold += len(gold_set)
        self._num_predicted += len(predicted_set)

    def recall(self):
        return maybe_divide(self._num_correct, self._num_gold)

    def precision(self):
        return maybe_divide(self._num_correct, self._num_predicted)

    def metrics(self):
        recall = self.recall()
        precision = self.precision()
        f1 = maybe_divide(2 * recall * precision, precision + recall)
        return recall, precision, f1


class EmbeddingDictionary(object):
  def __init__(self, info, normalize=True, maybe_cache=None):
    self._size = info["size"]
    self._normalize = normalize
    self._path = info["path"]
    if maybe_cache is not None and maybe_cache._path == self._path:
      assert self._size == maybe_cache._size
      self._embeddings = maybe_cache._embeddings
    else:
      self._embeddings = self.load_embedding_dict(self._path)

  @property
  def size(self):
    return self._size

  def load_embedding_dict(self, path):
    print("Loading word embeddings from {}...".format(path))
    default_embedding = np.zeros(self.size)
    embedding_dict = collections.defaultdict(lambda:default_embedding)
    if len(path) > 0:
      vocab_size = None
      with open(path) as f:
        for i, line in enumerate(f.readlines()):                  
          word_end = line.find("\t")
          word = line[:word_end]
          embedding = np.fromstring(line[word_end + 1:], np.float32, sep=",")
          #print(word)
          #print(embedding)          
          assert len(embedding) == self.size
          embedding_dict[word] = embedding
      if vocab_size is not None:
        assert vocab_size == len(embedding_dict)
      print("Done loading word embeddings.")
    return embedding_dict

  def __getitem__(self, key):
    embedding = self._embeddings[key]
    if self._normalize:
      embedding = self.normalize(embedding)
    return embedding

  def normalize(self, v):
    norm = np.linalg.norm(v)
    if norm > 0:
      return v / norm
    else:
      return v

class CustomLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, batch_size, dropout,contxt_embsize, maxkb,config,layer):
        self._num_units = num_units
        self._dropout = dropout
        
        self._contxt_embsize = contxt_embsize
        self._maxkb = maxkb
        self._config = config
        self._layer = layer
    
        
        self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), dropout)
        
        self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)    
        self._initializer1 = self._block_orthonormal_initializer([self.output_size]*1)
        self._initializer2 = self._block_orthonormal_initializer([self.output_size]*2)
        
        
        initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size])
        initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size])
        self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

    @property
    def output_size(self):
        return self._num_units

    @property
    def initial_state(self):
        return self._initial_state

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"      
            c, h = state
            h *= self._dropout_mask
          
            if self._config["noKB"]:
                
                inxs=inputs[:, :self._contxt_embsize]
                concat = projection(tf.concat([inxs, h], 1), 3 * self.output_size, initializer=self._initializer)
                i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
              
            if self._config["average_KB"]:
                
                inxs = inputs[:, :self._contxt_embsize]
                kb = inputs[:, self._contxt_embsize:]
                
                          
                with tf.variable_scope("j"):
                    j = projection(tf.concat([inxs, h], 1), 1 * self.output_size,initializer=self._initializer1)  # bsize,200
                with tf.variable_scope("io"):
                    concat = projection(tf.concat([inputs, h], 1), 2 * self.output_size, initializer=self._initializer2)  # bsize,400
    
                i, o = tf.split(concat, num_or_size_splits=2, axis=1)
          
            if self._config["attention_KB"]: 
                          
                inxs=inputs[:,:self._contxt_embsize]   
                with tf.variable_scope("gj"):
                    concat1 = projection(tf.concat([inxs, h], 1), 2 * self.output_size, initializer=self._initializer2)
                g, j = tf.split(concat1, num_or_size_splits=2,axis=1)  # g batchsize,200  
                
               
                if self._layer>0:
                    inputsa=inputs
                          
                else: 
                #print("########")                         
                    kb = inputs[:, self._contxt_embsize:]   #batchsize,maxkb*200
                    
                    #print("########") 
                    #print(self._maxkb)
                    #print(kb.get_shape())
                    
                    b_size = shape(kb, 0)  # batchsize
                    
                    kb_size = 200
                    kb = tf.reshape(kb, [-1, kb_size])     # batchsize*maxkb,200
                    kb_attention_v = tf.to_float(kb, name='ToFloat')  # batchsize*maxkb,200
                      
                    kb_attention_q = tf.reshape(g, [-1, 1, 200])  # batchsize,1,200
                    kb_attention_v = tf.reshape(kb_attention_v, [-1, 200, self._maxkb])  # batchsize,200,maxkb
                    
                   
                    
                    kb_attention_input = tf.matmul(kb_attention_q, kb_attention_v)  # batchsize,maxkb
                    
                  # attention
                    with tf.variable_scope("kb_attention_scores"):
                        self.kb_attention_scores = projection(kb_attention_input, kb_attention_input.get_shape().as_list()[2])  # batchsize,maxkb
                        
                    
                    kb_attention_score_softmax = tf.nn.softmax(self.kb_attention_scores, 2)  # batchsize,maxkb
                    kb_attention_emb = tf.reshape(kb_attention_score_softmax,[-1, self._maxkb, 1]) * tf.reshape(kb_attention_v, [-1, self._maxkb, 200])# batchsize,maxkb,200
    
                    kb_attention_emb=tf.reduce_sum(kb_attention_emb, 1)
                    #kb_attention_emb = tf.reshape(kb_attention_emb, [b_size, kb_attention_emb.get_shape().as_list()[1]*200])  # maxkb*batchsize,200                 
     
                    inputsa = tf.concat([inxs, kb_attention_emb], 1)
    
                with tf.variable_scope("io"):
                    concat2 = projection(tf.concat([inputsa, h], 1), 2 * self.output_size, initializer=self._initializer2)
                i, o = tf.split(concat2, num_or_size_splits=2, axis=1)  

            i = tf.sigmoid(i)
            new_c = (1 - i) * c  + i * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            
            return new_h, new_state

    def _orthonormal_initializer(self, scale=1.0):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
            M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
            Q1, R1 = np.linalg.qr(M1)
            Q2, R2 = np.linalg.qr(M2)
            Q1 = Q1 * np.sign(np.diag(R1))
            Q2 = Q2 * np.sign(np.diag(R2))
            n_min = min(shape[0], shape[1])
            params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
            return params
        return _initializer

    def _block_orthonormal_initializer(self, output_sizes):
        def _initializer(shape, dtype=np.float32, partition_info=None):
            assert len(shape) == 2
            assert sum(output_sizes) == shape[1]
            initializer = self._orthonormal_initializer()
            params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
            return params
        return _initializer
