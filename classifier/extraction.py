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

"""Extract embeddings from the model.

This file extracts embeddings and labels for all samples stored in the given TFRecord files and 
dumps the output to a json file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time

import json

import numpy as np
import tensorflow as tf

import configuration
import classifier_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("extraction_dir", "", "Directory to write extracted embeddings.")
tf.flags.DEFINE_string("extraction_file", "extraction", "Name of output file.")



tf.logging.set_verbosity(tf.logging.INFO)

def extract_all(sess, model, dump_file):
  """Computes perplexity-per-word over the evaluation dataset.

  Summaries and perplexity-per-word are written out to the eval directory.

  Args:
    sess: Session object.
    model: Instance of ShowAndTellModel; the model to evaluate.
    dump_file: Reference to file, where evaluation results should be stored
  """

  all_mi_labels = []
  all_sc_labels = []
  all_article_embeddings = []
  try:
    while True:
      mi_labels, sc_labels, article_embeddings = sess.run([
        model.mi_labels,
        model.sc_labels,
        model.article_embeddings,
      ])
      
      all_mi_labels += mi_labels.tolist()
      all_sc_labels += sc_labels.tolist()
      all_article_embeddings += article_embeddings.tolist()
      
      if not len(all_mi_labels) % 64: 
        tf.logging.info("Extracted embeddings for %d samples.", len(all_mi_labels))
      
  except tf.errors.OutOfRangeError as e:
    tf.logging.info("Finished extraction of %d samples", len(all_mi_labels))
    
  dump_arr = []
  
  for i in range(len(all_mi_labels)):
    dump_obj = {}
  
    mi = all_mi_labels[i]
    sc = all_sc_labels[i]
    ae = all_article_embeddings[i]
    
    dump_obj['mi'] = mi
    dump_obj['sc'] = sc
    dump_obj['ae'] = ae
    
    dump_arr.append(dump_obj)
      
  dump_str = json.dumps(dump_arr)
  dump_file.write(dump_str + '\n')
  
  tf.logging.info("Finished extraction")


def run_once(model, saver, dump_file, init_op):
  """Evaluates the latest model checkpoint.

  Args:
    model: Instance of ShowAndTellModel; the model to evaluate.
    saver: Instance of tf.train.Saver for restoring model Variables.
    dump_file: Reference to file, where evaluation results should be stored
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
      extract_all(
          sess=sess,
          model=model,
          dump_file=dump_file)
    except Exception as e:  # pylint: disable=broad-except
      tf.logging.error("Extraction failed.")
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def run():
  """Runs evaluation in a loop, and logs summaries to TensorBoard."""
  # Create the evaluation directory if it doesn't exist.
  extraction_dir = FLAGS.extraction_dir
  if not tf.gfile.IsDirectory(extraction_dir):
    tf.logging.info("Creating extraction directory: %s", extraction_dir)
    tf.gfile.MakeDirs(extraction_dir)

  # generate eval dump file
  dump_file = open(os.path.join(extraction_dir, FLAGS.extraction_file + '.json'), 'w')
    
  g = tf.Graph()
  with g.as_default():
    # Build the model for evaluation.
    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern
    model = classifier_model.Classifier(model_config, mode="extract")
    model.build()
    
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    # Create the Saver to restore model Variables.
    saver = tf.train.Saver()

    g.finalize()

    # Run a new evaluation run every eval_interval_secs.
    try:
      start = time.time()
      tf.logging.info("Starting extraction at " + time.strftime(
          "%Y-%m-%d-%H:%M:%S", time.localtime()))
      run_once(model, saver, dump_file, init_op)

    except KeyboardInterrupt:
      dump_file.close()


def main(unused_argv):
  assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
  assert FLAGS.extraction_dir, "--extraction_dir is required"
  run()


if __name__ == "__main__":
  tf.app.run()
