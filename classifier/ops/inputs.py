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

"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def parse_sequence_example(serialized, image_feature, sentences_feature, sentence_length, mi_feature, sc_feature):
  """Parses a tensorflow.SequenceExample into an image and text plus there corresponding labels.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.
    image_feature: Name of SequenceExample context feature containing image
      data.
    sentences_feature_name: Name of SequenceExample feature list containing integer
      sentences. (Sequence of fixed size feature vectors)
    sentence_length: Length of a single sentence in the sentence sequence (which is a fixed length feature vector)
    mi_feature: Name of SequenceExample context feature containing the mutual information label
    sc_feature: Name of SequenceExample context feature containing the semantic correlation label

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    text: An array containing a 2-D uint64 Tensor with dynamically first dimension. 
      The second dimension represents the tokens in each sentence.
    mi_lable: The mutual information label
    sc_label: The semantic correlation label
  """
  context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features={
          image_feature: tf.FixedLenFeature([], dtype=tf.string),
          mi_feature: tf.FixedLenFeature([], dtype=tf.int64),
          sc_feature: tf.FixedLenFeature([], dtype=tf.float32)
      },
      sequence_features={
          sentences_feature: tf.FixedLenSequenceFeature([sentence_length], dtype=tf.int64),
      })

  encoded_image = context[image_feature]
  sentences = sequence[sentences_feature]
  mi_label = context[mi_feature]
  sc_label = context[sc_feature]
  
  return encoded_image, sentences, mi_label, sc_label


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue",
                        mode=""):
  """Prefetches string values from disk into an input queue.

  In training the capacity of the queue is important because a larger queue
  means better mixing of training examples between shards. The minimum number of
  values kept in the queue is values_per_shard * input_queue_capacity_factor,
  where input_queue_memory factor should be chosen to trade-off better mixing
  with memory usage.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        /tmp/train_data-?????-of-00100).
    is_training: Boolean; whether prefetching for training or eval.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.
    mode: The current mode the model is running in. If mode is extract,
      then the queue runs ones through all samples and then throws an error.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern)) # tf.gfile.Glob returns a list of files matching the given pattern
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  num_epochs = None
  if mode == 'extract':
    num_epochs = 1
    
  if is_training:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size # the extra capacity allows the Queue to have enough time to enqueue new samples, since it should always hold at least min_queue_examples
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name, num_epochs=num_epochs)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue) # the reader will automatically dequeue the next filename as soon as it finished reading the current one
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops)) # this op will start the actual threads as soon as the session runs
  tf.summary.scalar(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue


def batch_with_dynamic_pad(images_and_texts,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):	
  """Batches input images, texts and their labels.

  This function generates an input sequence. Input sequences
  are batched and padded up to the maximum length of sequences
  in the batch. A mask is created to distinguish real sentences from padding sentences.

  Example:
    Actual senctences in the batch ('-' denotes padded sentence):
      [
        [ 1 2 5 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]

    input_seqs:
      [
        [ 1 2 5 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]

    mask:
      [
        [ 1 1 1 1 1 ],
        [ 1 1 1 1 0 ],
        [ 1 1 1 0 0 ],
      ]

  Args:
    images_and_texts: A list of pairs [image, texts, mi, sc], where image is a
      Tensor of shape [height, width, channels] and texts is a list of 2-D Tensors of
      any length. mi and sc are each scalar values. Each pair will be processed and 
      added to the queue in a separate thread.
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add text length summaries.

  Returns:
    images: A Tensor of shape [batch_size, height, width, channels].
    input_seqs: An int32 Tensor of shape [batch_size, padded_length, sentence_length].
    mi_labels: An int64 Tensor of shape [batch_size, padded_length].
    sc_labels: An float32 Tensor of shape [batch_size, padded_length].
    mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
  """
  enqueue_list = []
  for image, text, mi, sc in images_and_texts:
    text_length = tf.shape(text)[0]
    sentence_length = tf.shape(text)[1]
    
    input_length = text_length # create a tensor with 0 dims that contains caption_length-1

    input_seq = tf.slice(text, [0,0], [input_length, sentence_length])
    indicator = tf.ones([input_length], dtype=tf.int32)
    enqueue_list.append([image, input_seq, mi, sc, indicator])

  images, input_seqs, mi_labels, sc_labels, mask = tf.train.batch_join(
      enqueue_list,
      batch_size=batch_size,
      capacity=queue_capacity,
      dynamic_pad=True,
      name="batch_and_pad")

  if add_summaries:
    lengths = tf.add(tf.reduce_sum(mask, 1), 1)
    tf.summary.scalar("text_length/batch_min", tf.reduce_min(lengths))
    tf.summary.scalar("text_length/batch_max", tf.reduce_max(lengths))
    tf.summary.scalar("text_length/batch_mean", tf.reduce_mean(lengths))

  return images, input_seqs, mi_labels, sc_labels, mask
