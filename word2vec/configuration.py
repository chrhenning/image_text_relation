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

"""Word2vec model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""
    # File pattern of sharded TFRecord file containing SequenceExample protos.
    # Must be provided in training and evaluation modes.
    self.input_file_pattern = None

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.
    self.values_per_input_shard = 393
    # Minimum number of shards to keep in the input queue.
    self.input_queue_capacity_factor = 2
    # Number of threads for prefetching SequenceExample protos.
    self.num_input_reader_threads = 1
    
    # Approximate number of tokens per sequence_example. Used to ensure sufficient
    # mixing between examples in training.
    self.tokens_per_sequence_example = 50*40
    # Minimum number of sequence_examples to keep in samples queue.
    self.sample_queue_capacity_factor = 2

    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    self.vocab_size = 12600

    # Number of threads for image preprocessing. Should be a multiple of 2.
    self.num_preprocess_threads = 4

    # Batch size.
    self.batch_size = 128

    # Scale used to initialize model variables.
    self.initializer_scale = 1

    # size of word embedding vectors.
    self.embedding_size = 300

    # How many words to consider left and right.
    # [(skip_window context words) center_word (skip_window context words)]
    self.skip_window = 2      
    # How many samples should be generated for a center word?
    # How many words from the context should be taken?
    self.num_skips = 4 
    
    # Number of negative samples used by the NCE loss
    self.num_neg_samples = 64 


class TrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    self.num_examples_per_epoch = 190202*40

    # Optimizer for training the model.
    self.optimizer = "SGD"

    # Learning rate for the initial phase of training.
    self.initial_learning_rate = 1.0
    self.learning_rate_decay_factor = 0.5
    self.num_epochs_per_decay = 1.0

    # Learning rate when fine tuning the model parameters.
    self.finetuning_learning_rate = 0.0005

    # If not None, clip gradients to this value.
    self.clip_gradients = 5.0

    # How many model checkpoints to keep.
    self.max_checkpoints_to_keep = 5
