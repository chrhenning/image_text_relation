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
It implements an Classifier to judge about the mustual information
and semantic correlation of image text pairs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from ops import image_embedding
from ops import image_processing
from ops import inputs as input_ops

import numpy as np

class Classifier(object):
  """
  Implementation based on:
  Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode, train_encoder=False, use_pretrained_ae=True):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_encoder: Whether the encoder submodel variables are trainable.
      use_pretrained_ae: Whether to initialize from a pretrained encoder network.
    """
    assert mode in ["train", "eval", "extract", "prediction"]
    self.config = config
    self.mode = mode
    self.train_encoder = train_encoder
    self.train_inception = train_encoder
    self.use_pretrained_ae = use_pretrained_ae
    

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
    
    # A float32 scalar Tensor; the loss of the mutual information prediction.
    self.mi_loss = None
    
    # A float32 scalar Tensor; the loss of the semantic correlation prediction.
    self.sc_loss = None

    # A float32 Tensor with shape [batch_size * num_mi_labels].
    # Contains the unscaled likelihood of having a label
    self.mi_logits  = None
    
    # In case of considering semantic correlation prediction as multiclass problem:
    #   A float32 Tensor with shape [batch_size * num_sc_labels].
    # In case of considering semantic correlation prediction as a regression problem:
    #   A float32 Tensor with shape [batch_size].
    self.sc_logits  = None

    # Collection of variables from the autoencoder encoder network/model.
    self.autoencoder_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None
    
    # List of variables that should be trained or None, if all should be trained
    self.variables_to_train = None

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
    # in prediction mode, we use a batch size of one
    batch_size = self.config.batch_size
    
    if self.mode == "prediction":
      batch_size = 1
      
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed") # shape: scalar value

      #image_fn_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_fn_feed")
      
      #image_filename_queue = tf.train.string_input_producer([image_fn_feed]) #  list of files to read
   
      #reader = tf.WholeFileReader()
      #_, image_feed = reader.read(image_filename_queue)
      
      
      text_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None, self.config.sentence_length],  # shape 2D tensor - variable size (first dimension sentence sequence, second dimension token sequence (actually fixed size))
                                  name="text_feed")
      
      # arbitrary labels (not used)
      mi_label = tf.constant(-1, dtype=tf.int64) 
      sc_label = tf.constant(-1.0, dtype=tf.float32)                        

      image = self.process_image(image_feed)

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(text_feed, 0) 
      mi_labels = tf.expand_dims(mi_label, 0)
      sc_labels = tf.expand_dims(sc_label, 0)
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
          num_reader_threads=self.config.num_input_reader_threads,
          mode=self.mode)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_texts = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, text, mi, sc = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            sentences_feature=self.config.sentences_feature_name,
            sentence_length=self.config.sentence_length,
            mi_feature=self.config.mi_feature_name,
            sc_feature=self.config.sc_feature_name)
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_texts.append([image, text, mi, sc])

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        batch_size)
      images, input_seqs, mi_labels, sc_labels, input_mask = (
          input_ops.batch_with_dynamic_pad(images_and_texts,
                                           batch_size=batch_size,
                                           queue_capacity=queue_capacity))
        
    #print('Shapes')                  
    #print('Shape images: ' + str(images.get_shape()))
    #print('Shape input_seqs: ' + str(input_seqs.get_shape()))     
    #print('Shape input_mask: ' + str(input_mask.get_shape()))                                             

    self.images = images
    self.input_seqs = input_seqs
    if self.mode == "prediction":
      self.mi_labels = None
      self.sc_labels = None
    else:
      self.mi_labels = mi_labels
      self.sc_labels = sc_labels
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
          initializer=self.initializer)
          
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
    
    # All variables up until this point are shared with the autoencoder. So these are the variables
    # (the whole encoder network) that we want to restore/share.
    self.autoencoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      
  def build_prediction_model(self):
    """Builds the prediction model. Thus, it infers the mi resp. sc labels

    Inputs:
      self.article_embeddings
      self.mi_labels
      self.sc_labels

    Outputs:
      self.total_loss
    """
    batch_size=self.image_embeddings.get_shape()[0]
    
    
    with tf.variable_scope("pred_layer_1", initializer=self.initializer) as pred_scope_1:
    # We use a simple three layer fully-connected network for the actual prediction task
    # Hidden neuron sizes are randomly chosen 
    
      first_pred = tf.contrib.layers.fully_connected(
        inputs=self.article_embeddings,
        num_outputs=32,
        activation_fn=tf.nn.relu,
        weights_initializer=self.initializer,
        scope=pred_scope_1)
        
      if self.mode == "train":
          # to avoid overfitting we use dropout for all fully connected layers
          first_pred = tf.nn.dropout(first_pred, self.config.dropout_keep_prob_classifier)
    '''      
           
    with tf.variable_scope("pred_layer_2", initializer=self.initializer) as pred_scope_2:
      second_pred = tf.contrib.layers.fully_connected(
        inputs=first_pred,
        num_outputs=16,
        activation_fn=tf.nn.relu,
        weights_initializer=self.initializer,
        scope=pred_scope_2)
        
      if self.mode == "train":
          # to avoid overfitting we use dropout for all fully connected layers
          second_pred = tf.nn.dropout(second_pred, self.config.dropout_keep_prob_classifier)
    '''
    
    second_pred = first_pred
    
    ################################################
    # Predict Mutual Information
    ################################################
    
    with tf.variable_scope("predict_mi", initializer=self.initializer) as mi_scope:
      mi_logits = None
      mi_prediction = None
    
      if self.config.mi_is_multiclass_problem:
        mi_logits = tf.contrib.layers.fully_connected(
              inputs=second_pred,
              num_outputs=self.config.num_mi_labels,
              activation_fn=tf.nn.relu, #None, # linear activation 
              weights_initializer=self.initializer,
              scope=mi_scope)
                    
        if self.mode != "prediction":  
          # Compute loss
          mi_loss = 0.0
          
          # Do not punish all misclassifications equally.
          # Use the label distances defined in the config instead.
          if self.config.use_distance_aware_loss:
            # compute softmax to get probability of label l: p(l)
            mi_logits = tf.nn.softmax(mi_logits)
            
            # This part computes the Distance Aware loss function from section 4.4.1 of the thesis.
            # The loss allows to treat misclassifications differently based on a predefined
            # similarity metric defined between labels.
            # D is the symmetric matrix that defines the similarity d(l,t) for a label l
            # and a prediction t. The correct t is not known, only a softmax evidence for
            # all possible t. Therefore, we consider a whole column of D (corresponding to the correct
            # label l) and multiply this column by the softmax output to compute the loss value.        
            D = tf.constant(self.config.mi_label_distances)
            indices = tf.expand_dims(self.mi_labels, 1)
            d = tf.gather_nd(D, indices) # contains the l-th column of D for each batch
            
            mi_loss = tf.reduce_sum(d*mi_logits) # d(l,t) * p(t) for all t
      
          # use cross entropy loss
          else:
            # Applies softmax to the unscaled inputs (logits) and then computes the soft-entropy loss: H(p,q) = - sum p(x) * log q(x)
            mi_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(mi_logits, self.mi_labels) 

            mi_loss = tf.reduce_mean(mi_losses, name="mi_loss")  
            
          tf.contrib.losses.add_loss(mi_loss)
        
      else:
    
        # Consider the task as a regression problem and reduce its quadratic loss
        mi_prediction = tf.contrib.layers.fully_connected(
              inputs=second_pred,
              num_outputs=1,
              activation_fn=None, # linear activation 
              weights_initializer=self.initializer,
              scope=mi_scope)
                    
        if self.mode != "prediction":  
          mi_loss = tf.reduce_mean(tf.pow(mi_prediction - tf.to_float(self.mi_labels), 2))
          tf.contrib.losses.add_loss(mi_loss)
    
    ################################################
    # Predict Semantic Correlation
    ################################################
    
    with tf.variable_scope("predict_sc", initializer=self.initializer) as sc_scope:
      sc_logits = None
      sc_prediction = None
      
      if self.config.sc_is_multiclass_problem:
        
        # Consider prediction of semantic correlation as a multiclass problem
        sc_logits = tf.contrib.layers.fully_connected(
              inputs=second_pred,
              num_outputs=self.config.num_sc_labels,
              activation_fn=tf.nn.relu, #None, # linear activation 
              weights_initializer=self.initializer,
              scope=sc_scope)
              
        if self.mode != "prediction":                                  
          # compute sc multiclass labels
          # scale to [0,1,2,3,4]
          multiclass_labels = tf.to_int64(self.sc_labels * 2 + 2)

          # Compute loss
          sc_loss = 0.0
          
          # Do not punish all misclassifications equally.
          # Use the label distances defined in the config instead.
          if self.config.use_distance_aware_loss:
            # compute softmax to get probability of label l: p(l)
            sc_logits = tf.nn.softmax(sc_logits)
            
            # see comment above for distance aware MI loss
            D = tf.constant(self.config.sc_label_distances)
            indices = tf.expand_dims(multiclass_labels, 1)
            d = tf.gather_nd(D, indices) # contains the l-th column of D for each batch
            
            sc_loss = tf.reduce_sum(d*sc_logits) # d(l,t) * p(t) for all t
          
          # use cross entropy loss
          else:
            # Applies softmax to the unscaled inputs (logits) and then computes the soft-entropy loss: H(p,q) = - sum p(x) * log q(x)
            sc_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(sc_logits, multiclass_labels)  

            sc_loss = tf.reduce_mean(sc_losses, name="sc_loss") 
            
          tf.contrib.losses.add_loss(sc_loss)
      
      else:
    
        # Consider the task as a regression problem and reduce its quadratic loss
               
        sc_prediction = tf.contrib.layers.fully_connected(
              inputs=second_pred,
              num_outputs=1,
              activation_fn=None, # linear activation 
              weights_initializer=self.initializer,
              scope=sc_scope)
              
        if self.mode != "prediction":
          sc_loss = tf.reduce_mean(tf.pow(sc_prediction - self.sc_labels, 2))
          tf.contrib.losses.add_loss(sc_loss)

    
    if self.mode != "prediction":
      
      self.total_loss = tf.contrib.losses.get_total_loss()  
            
      self.mi_loss = mi_loss # used in evaluation
      self.sc_loss = sc_loss # used in evaluation
          
      # Add summaries.
      tf.summary.scalar("mi_loss", mi_loss)
      tf.summary.scalar("sc_loss", sc_loss)
      tf.summary.scalar("total_loss", self.total_loss)
      
      
    if self.config.mi_is_multiclass_problem:
      self.mi_logits = mi_logits # used in evaluation
    else:
      self.mi_logits = mi_prediction # used in evaluation
      
    if self.config.sc_is_multiclass_problem:
      self.sc_logits = sc_logits # used in evaluation
    else:
      self.sc_logits = sc_prediction # used in evaluation
      
    for var in tf.trainable_variables():
      #print(var)
      #print(var.name)
      tf.summary.histogram(var.op.name, var)


  def setup_encoder_initializer(self):
    """Sets up the function to restore AutoEncoder variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.autoencoder_variables)

      def restore_fn(sess):
        tf.logging.info("Restoring Autoencoder variables from checkpoint dir %s",
                        self.config.autoencoder_checkpoint_dir)
        saver.restore(sess, tf.train.latest_checkpoint(
                        self.config.autoencoder_checkpoint_dir))

      if self.use_pretrained_ae:
        self.init_fn = restore_fn
      else:
        self.init_fn = None

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def list_trainable_variables(self):
    """Store all variables that should be trainable into self.variables_to_train."""
    self.variables_to_train = None
    if not self.train_encoder:
      self.variables_to_train = list(set(tf.trainable_variables()) - set(self.autoencoder_variables))   

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.build_encoder()
    self.build_prediction_model()
    self.setup_encoder_initializer()
    self.setup_global_step()
    self.list_trainable_variables()
