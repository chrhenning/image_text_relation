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

"""Image-to-text model and training configurations."""

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

    # Image format ("jpeg" or "png").
    self.image_format = "jpeg"

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.
    self.values_per_input_shard = 393
    # Minimum number of shards to keep in the input queue.
    self.input_queue_capacity_factor = 2 
    # Number of threads for prefetching SequenceExample protos.
    self.num_input_reader_threads = 1

    # Name of the SequenceExample context feature containing image data.
    self.image_feature_name = "image/data"
    # Name of the SequenceExample feature list containing integer sentences.
    self.sentences_feature_name = "image/sentences_ids"
    
    # Length of a single sentence, Note, that sentences are fixed size
    self.sentence_length = 40 

    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    self.vocab_size = 12600

    # Number of threads for image preprocessing. Should be a multiple of 2.
    self.num_preprocess_threads = 4

    # Batch size.
    self.batch_size = 16

    # File containing an Inception v3 checkpoint to initialize the variables
    # of the Inception model. Must be provided when starting training for the
    # first time.
    self.inception_checkpoint_file = None
    
    # Path to checkpoint containing pretrained word embeddings.
    self.embedding_checkpoint_dir = None

    # Dimensions of Inception v3 input images.
    self.image_height = 300
    self.image_width = 300
    
    # there are three different methods to decode an image
    # 1. Linearly map article embedding onto large image, that is filtered through a CNN
    # 2. Linearly map article embedding onto thumbnail that is multiplied onto large image, that is filtered through a CNN
    # 3. Upsample (or resize) article embedding (that was mapped onto a thumbnail) stepwise, and filter image through CNN
    # NOTE: if you choose method 3, then image_height/width and thumbnail_height/width have to be 300 resp. 30.
    self.image_decoding_method = 3 
    
    # if image_decoding_method is 2 or 3
    # Thumbnail with and size. Image size must be a multiple of the thumbnail size.
    self.thumbnail_height = 30
    self.thumbnail_width = 30

    # Scale used to initialize model variables.
    self.initializer_scale = 0.08

    # Dimensionality of word embeddings
    self.word_embedding_size = 300
    # Dimensionality of sentence embeddings (Note, image embeddings will have the same size)  
    self.sentence_embedding_size = 600
    # Dimensionality of article embeddings
    self.article_embedding_size = 2400

    # If < 1.0, the dropout keep probability applied to LSTM and Fully-connected variables.
    self.dropout_keep_prob_encoder = 0.7
    self.dropout_keep_prob_decoder = 0.9
    
    # Reconsider the image after the whole article is read, as the image info 
    # might have gone lost in the RNN 
    self.reconsider_image = True
    
    # Since the predicted text loss is usually much higher, we scale the image loss by this factor to balance them
    #self.image_loss_factor = 10


class TrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    self.num_examples_per_epoch = 190202

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
