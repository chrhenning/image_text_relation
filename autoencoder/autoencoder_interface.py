#!/usr/bin/env python3
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

"""Predict.

Special script to retrieve MI and SC labels for top candidates for image-sentence retrieval task
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time

import json
import pickle

import numpy as np
import tensorflow as tf

import configuration
import classifier_model

import data.build_data as utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("prediction_dir", "", "Output directory.")
tf.flags.DEFINE_string("out_file", "predicted_candidates.json", "Name of output file.")
tf.flags.DEFINE_string("data_path", "", "Path to top candidates.")
tf.flags.DEFINE_string("coco_val_path", "", "Path to validation images of the coco dataset.")
tf.flags.DEFINE_string("vocab_file", "", "The file containing the vocabulary used to encode sentences.")
#tf.flags.DEFINE_integer("max_num_tokens", 40, "Maximum number of tokens in a sentence.")

tf.logging.set_verbosity(tf.logging.INFO)

MAX_NUM_TOKENS = 40

def predict_all(sess, model):
  """Predict MI and SC label with given model for all samples in file "data_path"
  Args:
    sess: Session object.
    model: Instance of ShowAndTellModel; the model to evaluate.
  """
  global MAX_NUM_TOKENS
  
  with open(FLAGS.data_path, 'r') as f:
    data = json.load(f)

  # image file pattern
  pattern = os.path.join(FLAGS.coco_val_path, 'COCO_val2014_%s.jpg')
  
  # Create a utility for decoding JPEG images to run sanity checks.
  #decoder = utils.ImageDecoder()
  
  # read vocabulary to encode sentences
  vocab = utils._read_vocab(FLAGS.vocab_file)

  count = 0
  
  for _, cand in data.items():
    imgid = cand['cocoid']
    
    img_fn = pattern % (str(imgid).zfill(12))
        
    if not os.path.isfile(img_fn):
      print("%s does not exist", img_fn)
      continue
    
    # read image
    with tf.gfile.GFile(img_fn.encode('utf8', 'surrogateescape'), "r") as f:
      encoded_image = f.read()
      
    # just a sanity check
    # FIXME graph is finalized, so we cannot run this sanity check at the moment
    #try:
    #  decoder.decode_jpeg(encoded_image)
    #except (tf.errors.InvalidArgumentError, AssertionError):
    #  print("Skipping file with invalid JPEG data: %s" % image.filename)
    #  continue
      
    sen = cand['sent']
    
    tokenized_sen = utils._process_sentence(sen, MAX_NUM_TOKENS)    
    
    token_ids = [vocab.word_to_id(word) for word in tokenized_sen]
    # pad sentence_ids if necessary
    token_ids += [0] * (MAX_NUM_TOKENS - len(token_ids)) 

    
    mi_logits, sc_logits, article_embeddings = sess.run([
      model.mi_logits,
      model.sc_logits,
      model.article_embeddings,
    ], feed_dict= { 'image_feed:0': encoded_image, 'text_feed:0': np.array([token_ids], dtype=np.int64)})
    
    if model.config.mi_is_multiclass_problem:
      mi_pred = mi_logits.argmax(axis=1).tolist()[0]
    else:
      mi_pred = np.squeeze(mi_logits).tolist()
          
    if model.config.sc_is_multiclass_problem:
      sc_pred = ((sc_logits.argmax(axis=1) - 2)/2.0).tolist()[0]
    else:
      sc_pred = np.squeeze(sc_logits).tolist()
    
    mi_label = mi_pred
    sc_label = sc_pred
    article_embedding = article_embeddings[0].tolist()
    
    cand['predicted_mi'] = mi_label
    cand['predicted_sc'] = sc_label
    cand['article_embedding'] = article_embedding

    count += 1
    if not count % 100: 
      tf.logging.info("Computed labels for %d samples.", count)
            
  # save predictions
  out_fn = os.path.join(FLAGS.prediction_dir, FLAGS.out_file)
    
  with open(out_fn, 'w') as f:
    data = json.dump(data, f)
    
  tf.logging.info("Saved results in %s.", out_fn)

  tf.logging.info("Finished prediction")


def run_once(model, saver, init_op):
  """Predict on the latest model checkpoint.

  Args:
    model: Instance of ShowAndTellModel; the model to evaluate.
    saver: Instance of tf.train.Saver for restoring model Variables.
  """
  if os.path.isdir(FLAGS.checkpoint_dir):
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    
    if not model_path:
      tf.logging.info("Skipping extraction. No checkpoint found in: %s",
                      FLAGS.checkpoint_dir)
      return

  else:
    # For instance, the user does not want to simply restore the latest checkpoint but a specific one (e.g. the one in the folder model/best)
    model_path = FLAGS.checkpoint_dir

  with tf.Session() as sess:
    
    sess.run(init_op)
        
    # Load model from checkpoint.
    tf.logging.info("Loading model from checkpoint: %s", model_path)
    saver.restore(sess, model_path)
    global_step = tf.train.global_step(sess, model.global_step.name)
    tf.logging.info("Successfully loaded %s at global step = %d.",
                    os.path.basename(model_path), global_step)
    

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Run evaluation on the latest checkpoint.
    try:
      predict_all(
          sess=sess,
          model=model)
    except Exception as e:  # pylint: disable=broad-except
      print(e)
      tf.logging.error("Prediction failed.")
      coord.request_stop(e)
      

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def run():
  """Runs evaluation in a loop, and logs summaries to TensorBoard."""
  # Create the evaluation directory if it doesn't exist.
  prediction_dir = FLAGS.prediction_dir
  if not tf.gfile.IsDirectory(prediction_dir):
    tf.logging.info("Creating prediction directory: %s", prediction_dir)
    tf.gfile.MakeDirs(prediction_dir)
    
  g = tf.Graph()
  with g.as_default():
    # Build the model for evaluation.
    model_config = configuration.ModelConfig()
    model = classifier_model.Classifier(model_config, mode="prediction")
    model.build()
    
    global MAX_NUM_TOKENS
    MAX_NUM_TOKENS = model_config.sentence_length
    
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    # Create the Saver to restore model Variables.
    saver = tf.train.Saver()

    g.finalize()

    # Run a new evaluation run every eval_interval_secs.
    try:
      start = time.time()
      tf.logging.info("Starting prediction at " + time.strftime(
          "%Y-%m-%d-%H:%M:%S", time.localtime()))
      run_once(model, saver, init_op)

    except KeyboardInterrupt:
      pass


def main(unused_argv):
  assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
  assert FLAGS.prediction_dir, "--prediction_dir is required"
  assert FLAGS.data_path, "--data_path is required"
  assert FLAGS.coco_val_path, "--coco_val_path is required"
  run()


if __name__ == "__main__":
  tf.app.run()
