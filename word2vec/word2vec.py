# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Notice, this file was modified by Christian Henning.
# The file or pieces of code in this file originate from:
# https://github.com/tensorflow/models/tree/master/im2txt

"""
This implementation derives from:
  Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  
Additionally, part of the code is taken from:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

This class implements a word embedding training. It used the word2vec skip-gram model as 
described here
https://www.tensorflow.org/tutorials/word2vec/

The trained word embeddings can then be used for other applications
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from ops import inputs as input_ops

import math

import numpy as np

class Word2Vec(object):
  """
  Implementation based on:
  Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # An int32 Tensor with shape [batch_size].
    self.center_words = None

    # An int32 Tensor with shape [batch_size,].
    self.context_words = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.center_words
      self.context_words
    """
    
    if self.mode == "train":
   
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      assert self.config.num_preprocess_threads % 2 == 0
      
      # create a queue for samples (center_word, context_word) from multiple sequence examples
      min_queue_examples = self.config.tokens_per_sequence_example * self.config.sample_queue_capacity_factor * self.config.num_skips
      capacity = min_queue_examples + 100 * self.config.batch_size # the extra capacity allows the Queue to have enough time to enqueue new samples, since it should always hold at least min_queue_examples
      samples_queue = tf.RandomShuffleQueue(
          capacity=capacity,
          min_after_dequeue=min_queue_examples,
          dtypes=[tf.int64],
          shapes=[2],
          name="random_samples_queue")
      
      enqueue_ops = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        sample_center_words, sample_context_words = input_ops.parse_sequence_example(
            serialized_sequence_example) 
        
        sample_center_words = tf.expand_dims(sample_center_words, 1)
        sample_context_words = tf.expand_dims(sample_context_words, 1)
        
        samples = tf.concat(1, [sample_center_words, sample_context_words])

        enqueue_ops.append(samples_queue.enqueue_many(samples))
      
      tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
          samples_queue, enqueue_ops)) # this op will start the actual threads as soon as the session runs
             
        
      random_samples = []
      for thread_id in range(self.config.num_preprocess_threads):
        sample = samples_queue.dequeue()
        
        random_samples.append(sample)          

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        self.config.batch_size)
                        
      center_words, context_words = (
          input_ops.batch_samples(random_samples,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity))
        
      #print('Shapes')                  
      #print('Shape center_words: ' + str(center_words.get_shape()))     
      #print('Shape context_words: ' + str(context_words.get_shape()))                                             

      self.center_words = center_words
      self.context_words = context_words
      
    else:
      valid_size = 16
      valid_dataset = tf.placeholder(tf.int64, [valid_size], name="valid_dataset")
      self.valid_dataset = valid_dataset

  def build_word2vec_model(self):
    """Builds the word2vec model.
    Code source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

    Inputs:
      self.center_words
      self.context_words

    Outputs:
      self.total_loss
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):

      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)

      # Construct the variables for the NCE loss
      nce_weights = tf.Variable(tf.truncated_normal([self.config.vocab_size, self.config.embedding_size],
                                                                stddev=1.0 / math.sqrt(self.config.embedding_size)))
      nce_biases = tf.Variable(tf.zeros([self.config.vocab_size]))
      
      if self.mode == "train":
        embed = tf.nn.embedding_lookup(embedding_map, self.center_words) 

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
            biases=nce_biases,
            labels=self.context_words,
            inputs=embed,
            num_sampled=self.config.num_neg_samples,
            num_classes=self.config.vocab_size))

        tf.contrib.losses.add_loss(loss)
        
        total_loss = tf.contrib.losses.get_total_loss()  
    
        # Add summaries.
        tf.scalar_summary("total_loss", total_loss)
        
        for var in tf.trainable_variables():
          #print(var)
          #print(var.name)
          tf.histogram_summary(var.op.name, var)

        self.total_loss = total_loss
          
      else:
        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_map), 1, keep_dims=True))
        self.normalized_embeddings = embedding_map / norm
        valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset)
        self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)


  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_word2vec_model()
    self.setup_global_step()
