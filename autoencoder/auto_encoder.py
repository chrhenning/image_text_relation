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
It implements an AutoEncoder for image text pairs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from ops import image_embedding
from ops import image_processing
from ops import inputs as input_ops

import numpy as np

class AutoEncoder(object):
  """
  Implementation based on:
  Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode, train_inception=False, train_embeddings=False):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception
    self.train_embeddings = train_embeddings

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, text_length, sentence_length].
    self.input_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, text_length].
    self.input_mask = None
    
    # An int32 0/1 Tensor with shape [batch_size, text_length, sentence_length].
    self.sentence_mask = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None
    
    # A float32 Tensor with shape [vocab_size, embedding_size].
    self.normed_embedding_map = None

    # A float32 Tensor with shape [batch_size, text_length, sentence_length, embedding_size].
    self.seq_embeddings = None
    
    # A float32 Tensor with shape [batch_size, embedding_size].
    self.article_embeddings = None
    
    # A int32 Tensor with shape [batch_size].
    self.sentence_sequence_length = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None
    
    # A float32 scalar Tensor; the loss of the image prediction.
    self.image_loss = None
    
    # A float32 scalar Tensor; the loss of the text prediction.
    self.text_loss = None

    # A float32 Tensor with shape [batch_size * text_length * sentence_length].
    self.image_quadratic_losses  = None
    
    # A float32 Tensor with shape [batch_size * text_length * sentence_length].
    self.text_cross_entropy_losses  = None
    
    # A float32 Tensor with shape [batch_size * text_length * sentence_length].
    self.text_cross_entropy_loss_weights  = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id,
                                          image_format=self.config.image_format)

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.input_mask (training and eval only)
    """
    batch_size = self.config.batch_size
    
    # In inference mode, we use a batch size of 1.
    if self.mode == "inference":

      batch_size = 1
      
      # In inference mode, images and texts are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed") # shape: scalar value    
      
      text_feed = tf.placeholder(dtype=tf.int64,
                                 shape=[None, self.config.sentence_length],  # shape 2D tensor - variable size (first dimension sentence sequence, second dimension token sequence (actually fixed size))
                                  name="text_feed")                      

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(text_feed, 0) 
      input_mask = tf.expand_dims(tf.constant([1], dtype=tf.int32) , 0)
      
    else:
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_texts = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, text = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            sentences_feature=self.config.sentences_feature_name,
            sentence_length=self.config.sentence_length)
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_texts.append([image, text])

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        batch_size)
      images, input_seqs, input_mask = (
          input_ops.batch_with_dynamic_pad(images_and_texts,
                                           batch_size=batch_size,
                                           queue_capacity=queue_capacity))
        
    #print('Shapes')                  
    #print('Shape images: ' + str(images.get_shape()))
    #print('Shape input_seqs: ' + str(input_seqs.get_shape()))     
    #print('Shape input_mask: ' + str(input_mask.get_shape()))                                             

    self.images = images
    self.input_seqs = input_seqs
    self.input_mask = input_mask

  def build_image_embeddings(self):
    """Builds the image model subgraph and generates image embeddings.

    Inputs:
      self.images

    Outputs:
      self.image_embeddings
    """
    inception_output = image_embedding.inception_v3(
        self.images,
        trainable=self.train_inception,
        is_training=self.is_training())
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    # Map inception output onto embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=inception_output,
          num_outputs=self.config.sentence_embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)
          
    if self.mode == "train":
      # to avoid overfitting we use dropout for all fully connected layers
      image_embeddings = tf.nn.dropout(image_embeddings, self.config.dropout_keep_prob_encoder)

    # Save the embedding size in the graph.
    tf.constant(self.config.sentence_embedding_size, name="image_embedding_size")

    self.image_embeddings = image_embeddings

  def build_seq_embeddings(self):
    """Builds the input sequence embeddings.

    Inputs:
      self.input_seqs

    Outputs:
      self.seq_embeddings
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.word_embedding_size],
          initializer=self.initializer,
          trainable=self.train_embeddings)
      self.embeddings_map = embedding_map
          
      # We need to store the normalized lookup table for efficient mapping of embedding vectors to closest words
      self.normed_embedding_map = tf.nn.l2_normalize(embedding_map, dim=1)
        
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs) 
      # seq_embeddings has the shape (batch_size, sequence_length, sentence_length, embedding_size)
      # meaning, for each index in input_seqs (with shape (batch_size, sequence_length, sentence_length)) it stores an embedding vector

    #print('Shape seq_embeddings: ' + str(seq_embeddings.get_shape()))

    self.seq_embeddings = seq_embeddings

  def build_encoder(self):
    """Builds the encoder model.

    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.input_mask (training and eval only)

    Outputs:
      self.article_embeddings
    """
    
    # some general variables concerning the current processed batch
    batch_size=self.image_embeddings.get_shape()[0]
    sentence_length = self.config.sentence_length # == self.seq_embeddings.get_shape()[2]
    max_text_length = tf.shape(self.seq_embeddings)[1] # maximum text length for this batch
    
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    
    # create an lstm cell that will process a sentence (a sequence of tokens)
    lstm_cell_sentences = tf.nn.rnn_cell.BasicLSTMCell(
        num_units=self.config.sentence_embedding_size, state_is_tuple=True) # num_units describes the size of the internal memory cell (but it is also the output size)
        
    # we also need an lstm cell that will process a sequence of sentences (a text)
    lstm_cell_text = tf.nn.rnn_cell.BasicLSTMCell(
        num_units=self.config.article_embedding_size, state_is_tuple=True)
        
    if self.mode == "train":
      # to avoid overfitting we use dropout for all lstm cells
      lstm_cell_sentences = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell_sentences,
          input_keep_prob=self.config.dropout_keep_prob_encoder,
          output_keep_prob=self.config.dropout_keep_prob_encoder)
      lstm_cell_text = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell_text,
          input_keep_prob=self.config.dropout_keep_prob_encoder,
          output_keep_prob=self.config.dropout_keep_prob_encoder)

    with tf.variable_scope("lstm_sentence_encode", initializer=self.initializer) as lstm_scope:
      # we use the image embedding only to feed the text lstm with image information
      # The sentences are initialized with a zero state
      
      # Set the initial LSTM state.
      initial_state_sentences = lstm_cell_sentences.zero_state(
          batch_size=batch_size, dtype=tf.float32)
      
      # At first, generate a mask for all sentences. 
      # This will allow us to specify the individual length of each sentence 
      # This lengths are fed into tf.nn.dynamic_rnn, which will produce zero outputs for 
      # all padded tokens.
      # Note, that self.input_seqs contains a zero for each padded token (zero is not in the vocabulary)
      zeros = tf.zeros_like(self.input_seqs)
      self.sentence_mask = tf.select(tf.greater(self.input_seqs, zeros) , tf.ones_like(self.input_seqs), zeros) # type int64

      #self.sentence_mask = tf.cast(self.sentence_mask, tf.int32)
      
      # In the following, we run a hierarchical approach:
      # Tokens of a sentence are mapped onto an embedding vector through lstm_cell_sentences
      # The resulting sentence embeddings are passed though lstm_cell_text to gather text embeddings
      
      # Since we have to generate an embedding for each sentence in a text, we need a loop somehow.
      # But the number of sentences in a text is dynamically determined for each batch (max_text_length).
      # Therefore, we cannot use unpack and a python loop. Instead we use the while_loop control method of TF.
      
      
      # The output of lstm_cell_sentences will be stored in this matrix, but only 
      # the lstm output of the last not padded word in a sentence
      lstm_outputs_sentences = tf.zeros(tf.pack([batch_size, max_text_length, self.config.sentence_embedding_size])) # tf.pack is a hotfix, since a normal array passing would not work as max_text_length is a tensor
      #lstm_outputs_sentences = tf.zeros([batch_size, max_text_length, self.config.embedding_size])
    
      # Allow the LSTM variables to be reused.
      #lstm_scope.reuse_variables()

      # now we compute the lstm outputs for each token sequence (sentence) in the while loop body
      def body(i,n,los):
        """Compute lstm outputs for sentences i (sentences with index i in text) of current batch.

        Inputs:
          i: control variable of loop (runs from 0 to n-1)
          n: max_text_length
          los: lstm_outputs_sentences

        Outputs:
          i: incremented
          n: unchanged
          los: input with updated values in index i of second dimension
        """
        # extract correct lstm input (i-th sentence from each batch)
        #es = tf.slice(self.seq_embeddings,[0,i,0,0],[batch_size, 1, sentence_length, self.config.word_embedding_size])
        es = tf.slice(self.seq_embeddings,tf.pack([0,i,0,0]),tf.pack([batch_size, 1, sentence_length, self.config.word_embedding_size]))
        es = tf.squeeze(es, axis=1) # get rid of sentence index dimension
        es = tf.reshape(es, tf.pack([batch_size, sentence_length, self.config.word_embedding_size])) # dirty hack, to ensure that shape is known (needed by further methods)

        # extract masks of sentences i
        sm = tf.slice(self.sentence_mask,tf.pack([0,i,0]),tf.pack([batch_size, 1, sentence_length]))
        sm = tf.squeeze(sm, axis=1)
        # compute sentence lengths
        sm = tf.reduce_sum(sm, 1)
        sm = tf.reshape(sm, tf.pack([batch_size])) # dirty hack, to ensure that shape is known

        # feed i-th sentences through lstm
        lstm_outputs_sentences_tmp, _ = tf.nn.dynamic_rnn(cell=lstm_cell_sentences,
                                                      inputs=es,
                                                      sequence_length=sm,
                                                      initial_state=initial_state_sentences,
                                                      dtype=tf.float32,
                                                      scope=lstm_scope)
        # lstm_outputs_sentences_tmp has shape (batch_size, sentence_length, sentence_embedding_size
        # lstm_outputs_sentences_tmp contains an output for each token in the sentences, but we are only interested in the 
        # output of the last token of a sentence
        
        # Now we extract only those outputs (output of last token, which is not a padded token) from lstm_outputs_sentences_tmp

        # sm contains the length of each sentence, meaning we can access the right output with the index (length - 1)
        # Note, that the actual masks where reduced to lengths in the above statements.
        sm = tf.sub(sm, 1) # sentence mask contains now the index of the last token in each sentence
        # Those sentence, that have zero tokens (padded sentences) have now an index of -1. We have to set them back to 0
        # which are simply zero outputs of the lstm
        zeros = tf.zeros_like(sm)
        sm = tf.select(tf.less(sm, zeros) , zeros, sm)

        # We use tf.gather_nd to extract the desired outputs from lstm_outputs_sentences_tmp.
        # Therefore, we have to produce the "indices" parameter of this method first.
        # The elements of the last dimension in this matrix determine the indices for gathering slices from lstm_outputs_sentences
        # Hence the innermost dimension must be a 2D vector: (batch, token) <- index of desired embedding in lstm_outputs_sentences
        # for sentence with index (batch, i) in self.seq_embeddings

        # We generate for each of the two indices a seperate matrix and concatenate them at the end
        sm = tf.expand_dims(sm, 1)
        sm = tf.cast(sm, dtype=tf.int32)

        # use tf.range to generate the equivalence of sm for batch indices
        #batch_indices = tf.range(0, batch_size)
        batch_indices = tf.constant(np.arange(int(batch_size)), dtype=tf.int32)
        batch_indices = tf.expand_dims(batch_indices, 1) 

        # then use tf.concat to generate the actual tensor, that can be used to gather the right outputs from lstm_outputs_sentences_tmp
        gather_indices = tf.concat(1, [batch_indices, sm])

        # now we can consider the elements (of the last dimension) of gather_indices as indices for the correct ouput
        lstm_outputs_sentences_tmp = tf.gather_nd(lstm_outputs_sentences_tmp, gather_indices)
        lstm_outputs_sentences_tmp = tf.expand_dims(lstm_outputs_sentences_tmp, 1) 

        # add the current output to our list of outputs
        los = tf.concat(1, [tf.slice(los, tf.pack([0,0,0]), tf.pack([batch_size, i, self.config.sentence_embedding_size])),
                                            lstm_outputs_sentences_tmp,
                                            tf.slice(los, tf.pack([0,i+1,0]), tf.pack([batch_size,n-i-1,self.config.sentence_embedding_size]))])
        
        return i+1,n,los

      def condition(i,n,los):
        """Break condition for while loop

        Inputs:
          i: control variable of loop (runs from 0 to n-1)
          n: max_text_length
          los: lstm_outputs_sentences

        Outputs:
          Ture, if body should be run.
        """

        return i < n

      result = tf.while_loop(condition, body, loop_vars=[0, max_text_length, lstm_outputs_sentences])
      lstm_outputs_sentences = result[2]       
        
    with tf.variable_scope("lstm_text_encode", initializer=self.initializer) as lstm_scope: 
            
      # Feed the image embeddings to set the initial LSTM state.
      zero_state_text = lstm_cell_text.zero_state(
          batch_size=batch_size, dtype=tf.float32)
      _, initial_state_text = lstm_cell_text(self.image_embeddings, zero_state_text)
    
      # Allow the LSTM variables to be reused.
      lstm_scope.reuse_variables()
            
      # lstm_outputs_sentences has now the last lstm output for each sentence in the batch (output of last unpadded token)
      # Its shape is (batch_size, max_text_length, sentence_embedding_size)
    
      # Now we use the sentence embeddings to generate text embeddings
      # Run the batch of sentence embeddings through the LSTM.
      self.sentence_sequence_length = tf.reduce_sum(self.input_mask, 1)
      lstm_outputs_text, _ = tf.nn.dynamic_rnn(cell=lstm_cell_text,
                                          inputs=lstm_outputs_sentences,
                                          sequence_length=self.sentence_sequence_length,
                                          initial_state=initial_state_text,
                                          dtype=tf.float32,
                                          scope=lstm_scope)
      # lstm_outputs_text has now the lstm output of each sentence_embedding,
      # where the output of the last unpadded sentence_embedding is considered as the text embedding.
      # Note, that we could also call it article embedding, since it comprises the information of the 
      # text and the image.
      # Its shape is (batch_size, max_text_length, article_embedding_size)

      # extract the text embedding from lstm_outputs_text
      
      # sequence_length contains the length of each text, meaning we can access the right output with the index (length - 1)
      last_sentence = tf.sub(self.sentence_sequence_length, 1) # sentence mask contains now the index of the last unpadded sentence in each text

      # We use tf.gather_nd to extract the desired outputs from lstm_outputs_text.
      # Therefore, we have to produce the "indices" parameter of this method first.
      # The elements of the last dimension in this matrix determine the indices for gathering slices from lstm_outputs_text
      # Hence the innermost dimension must be a 2D vector: (batch, sentence)

      # We generate for each of the two indices a seperate matrix and concatenate them at the end
      last_sentence = tf.expand_dims(last_sentence, 1)

      # use tf.range to generate the equivalence of sm for batch indices
      batch_indices = tf.range(0, batch_size)
      batch_indices = tf.expand_dims(batch_indices, 1) 

      # then use tf.concat to generate the actual tensor, that can be used to gather the right outputs from lstm_outputs_text
      gather_indices = tf.concat(1, [batch_indices, last_sentence])
      
      # now we can consider the elements (of the last dimension) of gather_indices as indices for the correct ouput
      self.article_embeddings = tf.gather_nd(lstm_outputs_text, gather_indices)
        
    # As the image information might have gone lost in the hierarchical rnn, the reader might reconsider it.
    if self.config.reconsider_image:
      with tf.variable_scope("reconsider_image", initializer=self.initializer, reuse=None) as reconsider_image_scope: 
        # concat current article embedding with image_embedding and map them through an fully connected layer onto a new embedding
        article_image_concat = tf.concat(1, [self.article_embeddings, self.image_embeddings])
        
        self.article_embeddings = tf.contrib.layers.fully_connected(
          inputs=article_image_concat,
          num_outputs=self.config.article_embedding_size,
          activation_fn=tf.nn.relu, #None, # linear activation 
          weights_initializer=self.initializer,
          scope=reconsider_image_scope)
          
        if self.mode == "train":
          # to avoid overfitting we use dropout for all fully connected layers
          self.article_embeddings = tf.nn.dropout(self.article_embeddings, self.config.dropout_keep_prob_encoder)
    
    # self.article_embeddings contains now the text/article embedding for each article in the batch
    # Its shape is (batch_size, article_embedding_size)
      
  def build_decoder(self):
    """Builds the decoder model.

    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.input_mask

    Outputs:
      self.total_loss
      self.image_quadratic_losses
      self.text_cross_entropy_losses
      self.text_cross_entropy_loss_weights
    """
    
    # some general variables concerning the current processed batch
    batch_size=int(self.image_embeddings.get_shape()[0])
    sentence_length = self.config.sentence_length # == self.seq_embeddings.get_shape()[2]
    max_text_length = tf.shape(self.seq_embeddings)[1] # maximum text length for this batch
    image_width = self.config.image_width
    image_height = self.config.image_height
    
    ################################################
    # At first, decode article embedding to text
    ################################################
    
    # create an lstm_cell that generates sequence of sentence embeddings from a given text embedding
    lstm_cell_text = tf.nn.rnn_cell.BasicLSTMCell(
        num_units=self.config.sentence_embedding_size, state_is_tuple=True)
    
    # create an lstm cell that generates a sequence of word embeddings from a given sentence embedding
    lstm_cell_sentences = tf.nn.rnn_cell.BasicLSTMCell(
        num_units=self.config.word_embedding_size, state_is_tuple=True)
        
    if self.mode == "train":
      # to avoid overfitting we use dropout for all lstm cells
      lstm_cell_sentences = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell_sentences,
          input_keep_prob=self.config.dropout_keep_prob_decoder,
          output_keep_prob=self.config.dropout_keep_prob_decoder)
      lstm_cell_text = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell_text,
          input_keep_prob=self.config.dropout_keep_prob_decoder,
          output_keep_prob=self.config.dropout_keep_prob_decoder)

    with tf.variable_scope("lstm_text_decode", initializer=self.initializer) as lstm_scope:
      
      # Set the initial LSTM state.
      initial_state_text = lstm_cell_text.zero_state(
          batch_size=batch_size, dtype=tf.float32)
      
      # generate lstm input (input will be the article embedding in each iteration and the previous state       
      lstm_text_input = tf.expand_dims(self.article_embeddings, 1)
      lstm_text_input = tf.tile(lstm_text_input, [1,max_text_length,1])
      
      # produce sentence embeddings
      lstm_outputs_text, _ = tf.nn.dynamic_rnn(cell=lstm_cell_text,
                                            inputs=lstm_text_input,
                                            sequence_length=self.sentence_sequence_length,
                                            initial_state=initial_state_text,
                                            dtype=tf.float32,
                                            scope=lstm_scope)
      # lstm_outputs_text has shape (batch_size, max_text_length, sentence_embedding_size), whereas the sentence embedding is a zero vector, if the sentence 
      # hasn't existed in input
      # Note, this model does not allow a variable sized sequence of sentence (it takes the same lengths as in the input)
      
    with tf.variable_scope("lstm_sentence_decode", initializer=self.initializer) as lstm_scope:
      # for each sentence embedding, generate a sequence of word embeddings    
      
      # Stack batches vertically. Each sentence is considered its own batch
      lstm_sentence_input_stacked = tf.reshape(lstm_outputs_text, [-1, self.config.sentence_embedding_size])
      sentence_mask_stacked = tf.reshape(self.sentence_mask, [-1, sentence_length])
      
      # Now we generate for each sentence (batch) a sequence.
      # Copy the sentence_embedding sentence_length times in order to use it as input to the next lstm.
      lstm_sentence_input_stacked = tf.expand_dims(lstm_sentence_input_stacked, 1)
      lstm_sentence_input_stacked = tf.tile(lstm_sentence_input_stacked, [1,sentence_length,1])
      
      sentence_lengths = tf.reduce_sum(sentence_mask_stacked, 1)
      
      initial_state_sentences = lstm_cell_sentences.zero_state(
            batch_size=tf.shape(sentence_lengths)[0], dtype=tf.float32)
      
      # Generate a token embedding for each for each token according to sentence_mask_stacked from the corresponding sentence embedding
      lstm_outputs_sentences, _ = tf.nn.dynamic_rnn(cell=lstm_cell_sentences,
                                              inputs=lstm_sentence_input_stacked,
                                              sequence_length=sentence_lengths,
                                              initial_state=initial_state_sentences,
                                              dtype=tf.float32,
                                              scope=lstm_scope)
      # lstm_outputs_sentences has now the shape (batch_size*max_text_length, sentence_length, word_embedding_size)
      
      # lstm_outputs_sentences is a list of sentences, where each sentence is a list of token embeddings
      # Now we have to figure out, which is the closest token in our vocabular according to this token embeddings
      # We compute this using the cosine similarity: http://stackoverflow.com/questions/37558899/efficiently-finding-closest-word-in-tensorflow-embedding
      
      # Stack outputs for each token
      lstm_outputs_sentences_stacked = tf.reshape(lstm_outputs_sentences, [-1, self.config.word_embedding_size])
      input_seqs_stacked = tf.reshape(self.input_mask, [-1])
      
      normed_predicted_token_embeddings = tf.nn.l2_normalize(lstm_outputs_sentences_stacked, dim=1)
      
      cosine_similarity = tf.matmul(normed_predicted_token_embeddings, tf.transpose(self.normed_embedding_map))
      
      ''' We can't use argmax, since we still need to backpropagate gradients. Therefore we use softmax.
      # Consider the token with the highest cosine similarity (between embedding vector and token embedding) as closest word
      predicted_tokens = tf.argmax(cosine_similarity, 1)
      
      # We compute the mean of the qudratic loss of token predictions as loss function.
      # Note, we want the system to output the exact same text.
      # But we do not allow the system to generate the length of the tokens by itself. 
      
      # Only consider those tokens, that are not padding words when computing the loss
      zeros = tf.zeros_like(predicted_tokens)
      predicted_tokens = tf.select(tf.greater(input_seqs_stacked, zeros) , tf.ones_like(predicted_tokens), zeros)
      
      # We cannot use tf.argmax directly, since it returns integers, but a loss expects floats.
      # This trick here doesn't work either, as we loose the track to the initial variables.
      predicted_tokens_f = tf.constant(np.arange(self.config.vocab_size), dtype=tf.float32)
      predicted_tokens_f = tf.gather(predicted_tokens_f, predicted_tokens)
      
      # Compute mean of quadratic loss of predictions.
      text_loss = tf.reduce_mean(tf.pow(tf.to_float(input_seqs_stacked) - predicted_tokens_f, 2)) 
      '''
      # We can cast inputs, as we do not have to backpropagate gradients that far
      weights = tf.to_float(input_seqs_stacked)

      # Compute losses.
      # Applies softmax to the unscaled inputs (logits) and then computes the soft-entropy loss: H(p,q) = - sum p(x) * log q(x)
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(cosine_similarity, input_seqs_stacked) 
      
      # the higher the cross entropy, the higher will be the loss (weighted sum)
      text_loss = tf.div(tf.reduce_sum(tf.mul(losses, weights)),
                        tf.reduce_sum(weights),
                        name="text_loss")

      tf.contrib.losses.add_loss(text_loss)
      
      if self.mode == 'inference':
        predicted_tokens = tf.argmax(cosine_similarity, 1)
      
        # Only consider those tokens, that are not padding words
        zeros = tf.zeros_like(predicted_tokens)
        predicted_tokens = tf.select(tf.greater(input_seqs_stacked, zeros) , predicted_tokens, zeros)
          
        self.decoded_text = predicted_tokens
                                   

    ################################################
    # Decode article embedding to image 
    ################################################
    with tf.variable_scope("image_decode", initializer=self.initializer) as conv_scope:
      # generate input for CNN from article embedding
      with tf.variable_scope("fully_connected1") as fc_scope:
          encoded_image = None
      
          # Linearly map the article embedding onto a tensor that is much larger (thus copies information).
          # So the whole information of an article embedding should be encoded in each region of this tensor,
          # such that a region-wise (convolutional net) algorithm can extract it to generate image regions.
          if self.config.image_decoding_method == 1:      
            encoded_image = tf.contrib.layers.fully_connected(
              inputs=self.article_embeddings,
              num_outputs=4*image_width*image_height,
              activation_fn=None, # linear activation 
              weights_initializer=self.initializer,
              scope=fc_scope)
              
            if self.mode == "train":
              # to avoid overfitting we use dropout for all fully connected layers
              encoded_image = tf.nn.dropout(encoded_image, self.config.dropout_keep_prob_decoder)

            # Reshape to huge image, in order to input the tensor into a CNN
            encoded_image = tf.reshape(encoded_image, [batch_size, 2*image_width, 2*image_height])
            
          # Linearly map the article embedding onto a thumbnail
          else:      
            assert(self.config.image_height % self.config.thumbnail_height == 0)
            assert(self.config.image_width % self.config.thumbnail_width == 0)
                      
            thumbnail = tf.contrib.layers.fully_connected(
              inputs=self.article_embeddings,
              num_outputs=self.config.thumbnail_height*self.config.thumbnail_width,
              activation_fn=None, # linear activation 
              weights_initializer=self.initializer,
              scope=fc_scope)
              
            if self.mode == "train":
              # to avoid overfitting we use dropout for all fully connected layers
              thumbnail = tf.nn.dropout(thumbnail, self.config.dropout_keep_prob_decoder)

            # Reshape to quadratic image, in order to input the tensor into a CNN
            encoded_image = tf.reshape(thumbnail, [batch_size, self.config.thumbnail_width, self.config.thumbnail_height])
              
            if self.config.image_decoding_method == 2: 
                            
              # copy thumbnail along horizontal axis
              encoded_image = tf.concat(1, [encoded_image for _ in range(2*int(self.config.image_width / self.config.thumbnail_width))])
              # copy thumbnail row along vertical axis
              encoded_image = tf.concat(2, [encoded_image for _ in range(2*int(self.config.image_height / self.config.thumbnail_height))])
              
            else:
              assert(self.config.image_decoding_method == 3)
              assert(self.config.image_width == 300 and self.config.image_height == 300)
              assert(self.config.thumbnail_width == 30 and self.config.thumbnail_height == 30)             
          
          encoded_image = tf.expand_dims(encoded_image, 3)
      
      if self.config.image_decoding_method == 1 or self.config.image_decoding_method == 2:     
        # A CNN layer to extract the visual information from the article embedding
        with tf.variable_scope("conv1") as conv_scope:
          W = tf.get_variable("weights", shape=[5,5,1,32], initializer=self.initializer) # 5x5 conv, 1 input, 32 outputs
          b = tf.get_variable("bias", shape=[32], initializer=self.initializer)
          partial_decoded_image = tf.nn.conv2d(encoded_image, W, [1, 1, 1, 1], padding='SAME')
          partial_decoded_image = tf.nn.bias_add(partial_decoded_image, b)
          partial_decoded_image = tf.nn.relu(partial_decoded_image, name='relu1')
          partial_decoded_image = tf.nn.max_pool(partial_decoded_image, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
  padding='SAME', name='pool1')
          partial_decoded_image = tf.nn.lrn(partial_decoded_image, name='norm1')

        # Linearly map the output of the last layer (visual information distributed in many layers) 
        # onto the final prediction
        with tf.variable_scope("conv2") as conv_scope:
          W = tf.get_variable("weights", shape=[5,5,32,3], initializer=self.initializer) # 5x5 conv, 32 input, 32 outputs
          b = tf.get_variable("bias", shape=[3], initializer=self.initializer)
          partial_decoded_image = tf.nn.conv2d(partial_decoded_image, W, [1, 1, 1, 1], padding='SAME')
          partial_decoded_image = tf.nn.bias_add(partial_decoded_image, b)

      else:
        with tf.variable_scope("upsample1") as upsample_scope:
          new_width = self.config.thumbnail_width * 2 # 60
          new_height = self.config.thumbnail_height * 2
          # upsample image, but do not distort content
          partial_decoded_image = tf.image.resize_images(encoded_image, [new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
      
        with tf.variable_scope("conv1") as conv_scope:
          W = tf.get_variable("weights", shape=[5,5,1,32], initializer=self.initializer) # 5x5 conv, 1 input, 32 outputs
          b = tf.get_variable("bias", shape=[32], initializer=self.initializer)
          partial_decoded_image = tf.nn.conv2d(partial_decoded_image, W, [1, 1, 1, 1], padding='SAME')
          partial_decoded_image = tf.nn.bias_add(partial_decoded_image, b)
          partial_decoded_image = tf.nn.relu(partial_decoded_image, name='relu1')
          #partial_decoded_image = tf.nn.max_pool(partial_decoded_image, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool1')
          partial_decoded_image = tf.nn.lrn(partial_decoded_image, name='norm1')
          
        with tf.variable_scope("upsample2") as upsample_scope:
          new_width = new_width * 2 # 120
          new_height = new_height * 2
          # upsample image, but do not distort content
          partial_decoded_image = tf.image.resize_images(partial_decoded_image, [new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
      
        with tf.variable_scope("conv2") as conv_scope:
          W = tf.get_variable("weights", shape=[5,5,32,8], initializer=self.initializer) # 5x5 conv, 32 input, 8 outputs
          b = tf.get_variable("bias", shape=[8], initializer=self.initializer)
          partial_decoded_image = tf.nn.conv2d(partial_decoded_image, W, [1, 1, 1, 1], padding='SAME')
          partial_decoded_image = tf.nn.bias_add(partial_decoded_image, b)
          partial_decoded_image = tf.nn.relu(partial_decoded_image, name='relu2')
          #partial_decoded_image = tf.nn.max_pool(partial_decoded_image, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool2')
          partial_decoded_image = tf.nn.lrn(partial_decoded_image, name='norm2')
          
        with tf.variable_scope("upsample3") as upsample_scope:
          new_width = int(new_width * 2.5) # 300
          new_height = int(new_height * 2.5)
          # upsample image, but do not distort content
          partial_decoded_image = tf.image.resize_images(partial_decoded_image, [new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
      
        with tf.variable_scope("conv3") as conv_scope:
          W = tf.get_variable("weights", shape=[5,5,8,3], initializer=self.initializer) # 5x5 conv, 8 input, 3 outputs
          b = tf.get_variable("bias", shape=[3], initializer=self.initializer)
          partial_decoded_image = tf.nn.conv2d(partial_decoded_image, W, [1, 1, 1, 1], padding='SAME')
          partial_decoded_image = tf.nn.bias_add(partial_decoded_image, b)
          #partial_decoded_image = tf.nn.relu(partial_decoded_image, name='relu3')
          #partial_decoded_image = tf.nn.max_pool(partial_decoded_image, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool3')
          #partial_decoded_image = tf.nn.lrn(partial_decoded_image, name='norm3')
      

      # Compute image loss.
      # This will be computed as the quadratic loss of the original image to the predicted image
      decoded_image = tf.reshape(partial_decoded_image, [batch_size, -1])
      orig_image = tf.reshape(self.images, [batch_size, -1])
      
      image_losses = tf.pow(decoded_image - orig_image, 2)
      image_loss = tf.reduce_mean(image_losses)
      # scale image_loss such that it is balanced with text_loss
      #image_loss = image_loss * self.config.image_loss_factor

      tf.contrib.losses.add_loss(image_loss)
    
    total_loss = tf.contrib.losses.get_total_loss()  
    
    self.image_loss = image_loss # used in evaluation
    self.text_loss = text_loss # used in evaluation
    self.image_quadratic_losses = image_losses # used in evaluation
    self.text_cross_entropy_losses = losses # used in evaluation
    self.text_cross_entropy_loss_weights = weights # used in evaluation
    
    self.decoded_image = partial_decoded_image
    
    # Add summaries.
    tf.summary.scalar("text_loss", text_loss)
    tf.summary.scalar("image_loss", image_loss)
    tf.summary.scalar("total_loss", total_loss)
    
    tf.summary.image("input_image", self.images, max_outputs=3, collections=None)
    tf.summary.image("decoded_image", partial_decoded_image, max_outputs=3, collections=None)
    
    for var in tf.trainable_variables():
      #print(var)
      #print(var.name)
      tf.summary.histogram(var.op.name, var)

    self.total_loss = total_loss

  def setup_inception_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver_inception = tf.train.Saver(self.inception_variables)
      
      # Restore word embeddings
      saver_embedding = tf.train.Saver([self.embeddings_map])

      def restore_fn(sess):
        tf.logging.info("Restoring Inception and word embedding variables from checkpoint file %s and dir %s"
                        % (self.config.inception_checkpoint_file, self.config.embedding_checkpoint_dir))
        saver_inception.restore(sess, self.config.inception_checkpoint_file)
        
        saver_embedding.restore(sess, tf.train.latest_checkpoint(self.config.embedding_checkpoint_dir))

      self.init_fn = restore_fn

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.build_encoder()
    self.build_decoder()
    self.setup_inception_initializer()
    self.setup_global_step()
