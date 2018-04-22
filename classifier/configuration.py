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

"""Classifier model and training configurations."""

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
    self.values_per_input_shard = 48
    # Minimum number of shards to keep in the input queue.
    self.input_queue_capacity_factor = 8
    # Number of threads for prefetching SequenceExample protos.
    self.num_input_reader_threads = 1

    # Name of the SequenceExample context feature containing image data.
    self.image_feature_name = "image/data"
    # Name of the SequenceExample feature list containing integer sentences.
    self.sentences_feature_name = "image/sentences_ids"
    # Name of the SequenceExample feature containing the mutual information annotation.
    self.mi_feature_name = "annotation/mi"
    # Name of the SequenceExample feature containing the semantic correlation annotation.
    self.sc_feature_name = "annotation/sc"
    
    # The number of labels used to classify mutual information
    #self.num_mi_labels = 8
    self.num_mi_labels = 5 # simplified MI labels (without labels 1, 6 and 7). Note, that the remaining labels have changed the numbering.
    # The number of labels used to classify semantic correlation
    self.num_sc_labels = 5
    # Whether MI prediction should be considered as a regression problem (qudratic loss to 
    # predicted value is minimized) or as a multiclass problem.
    self.mi_is_multiclass_problem = True 
    # Whether SC prediction should be considered as a regression problem (qudratic loss to 
    # predicted value is minimized) or as a multiclass problem.
    self.sc_is_multiclass_problem = True 
    # For multiclass problems you may either use cross entropy loss or
    # distance aware loss. Distance aware loss uses distance matrices,
    # that define the distance between to labels. Therefore, misclassifications
    # can be treated differently.
    # Note, it doesn't mak any difference when considering pure regression problems
    self.use_distance_aware_loss = False
    
    # Distances of Mutual Information labels (sysmmetric matrix)
    '''
    self.mi_label_distances = [
      [0.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.5, 0.5],
      [1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 0.5, 0.5],
      [0.8, 0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.8],
      [0.6, 0.4, 0.2, 0.0, 0.2, 0.4, 0.6, 0.6],
      [0.4, 0.6, 0.4, 0.2, 0.0, 0.2, 0.4, 0.4],
      [0.2, 0.8, 0.6, 0.4, 0.2, 0.0, 0.2, 0.2],
      [0.5, 0.5, 0.8, 0.6, 0.4, 0.2, 0.0, 1.0],
      [0.5, 0.5, 0.8, 0.6, 0.4, 0.2, 1.0, 0.0]
    ]
    '''
    # Distances of simplified Mutual Information labels (sysmmetric matrix).
    # Thus, labels 6 and 7 have been removed.
    self.mi_label_distances = [
      [ 0.0  , 0.25 , 0.5  , 0.75 , 1.0   ],
      [ 0.25 , 0.0  , 0.25 , 0.5  , 0.75  ],
      [ 0.5  , 0.25 , 0.0  , 0.25 , 0.5   ],
      [ 0.75 , 0.5  , 0.25 , 0.0  , 0.25  ],
      [ 1.0  , 0.75 , 0.5  , 0.25 , 0.0   ]
    ]

    # Distances of Semantic Correlation labels (sysmmetric matrix)
    self.sc_label_distances = [
      [0.0, 0.0, 1.0, 1.0, 1.0],
      [0.0, 0.0, 0.0, 1.0, 1.0],
      [1.0, 0.0, 0.0, 0.0, 1.0],
      [1.0, 1.0, 0.0, 0.0, 0.0],
      [1.0, 1.0, 1.0, 0.0, 0.0]
    ]
    
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

    # Dir containing an Autoencoder checkpoint to initialize the variables
    # of the encoder model. Must be provided when starting training for the
    # first time.
    self.autoencoder_checkpoint_dir = None

    # Dimensions of Inception v3 input images.
    self.image_height = 300
    self.image_width = 300

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
    self.dropout_keep_prob_classifier = 0.9
    
    # Reconsider the image after the whole article is read, as the image info 
    # might have gone lost in the RNN 
    self.reconsider_image = True


class TrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    self.num_examples_per_epoch = 521

    # Optimizer for training the model.
    self.optimizer = "SGD"

    # Learning rate for the initial phase of training.
    self.initial_learning_rate = 0.1
    self.learning_rate_decay_factor = 0.5
    self.num_epochs_per_decay = 8.0

    # Learning rate when fine tuning the model parameters.
    self.finetuning_learning_rate = 0.0005

    # If not None, clip gradients to this value.
    self.clip_gradients = 5.0

    # How many model checkpoints to keep.
    self.max_checkpoints_to_keep = 5
