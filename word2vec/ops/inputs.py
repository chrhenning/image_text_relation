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


def parse_sequence_example(serialized):
  """Parses a tensorflow.SequenceExample into a lif of word2vec samples generated from the current image-text pair.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.

  Returns:
    center_words: A scalar uint64 Tensor containing center word ids.
    context_words: A scalar uint64 Tensor containing context word ids.
  """
  context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features={
      },
      sequence_features={
          "word2vec/center_words": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "word2vec/context_words": tf.FixedLenSequenceFeature([], dtype=tf.int64)
      })

  center_words = sequence["word2vec/center_words"]
  context_words = sequence["word2vec/context_words"]
  
  return center_words, context_words


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
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
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue) # the reader will automatically dequeue the next filename as soon as it finished reading the current one
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops)) # this op will start the actual threads as soon as the session runs
  tf.scalar_summary(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue


def batch_samples(random_samples,
                           batch_size,
                           queue_capacity):	
  """Batches input samples.


  Args:
    random_samples: A list of tensors with shape (2),
      containing a center and context word
    batch_size: Batch size.
    queue_capacity: Queue capacity.

  Returns:
    center_words: An int32 Tensor of shape [batch_size].
    context_words: An int32 Tensor of shape [batch_size, ].
  """
  enqueue_list = []
  for sample in random_samples:
    center_word, context_word = tf.split(0, 2, sample)

    enqueue_list.append([center_word, context_word])

  center_words, context_words = tf.train.batch_join(
      enqueue_list,
      batch_size=batch_size,
      capacity=queue_capacity,
      dynamic_pad=False,
      name="batch")

  return tf.squeeze(center_words), context_words
