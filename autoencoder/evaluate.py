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

"""Evaluate the model.

This script should be run concurrently with training so that summaries show up
in TensorBoard.
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
import auto_encoder

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("eval_dir", "", "Directory to write event logs.")

tf.flags.DEFINE_integer("eval_interval_secs", 600,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples", 1024,
                        "Number of examples for evaluation.")

tf.flags.DEFINE_integer("min_global_step", 5000,
                        "Minimum global step to run evaluation.")

tf.logging.set_verbosity(tf.logging.INFO)


def evaluate_model(sess, model, global_step, summary_writer, summary_op, dump_file):
  """Computes perplexity-per-word over the evaluation dataset.

  Summaries and perplexity-per-word are written out to the eval directory.

  Args:
    sess: Session object.
    model: Instance of ShowAndTellModel; the model to evaluate.
    global_step: Integer; global step of the model checkpoint.
    summary_writer: Instance of SummaryWriter.
    summary_op: Op for generating model summaries.
    dump_file: Reference to file, where evaluation results should be stored
  """
  # for a sequence of words s_i, perplexity is defined as
  # 2^(-1/N sum_i^N log(p(s_i))
  # (The goal is to achieve a low perplexity)
  # -log p(s_i) can also be interpreted as the loss of a word (the smaller its probability the higher its loss)
  # this definition is used for the calculations in this method
  
  # Log model summaries on a single batch.
  summary_str = sess.run(summary_op)
  summary_writer.add_summary(summary_str, global_step)
  
  # Log evaluation into this dict
  curr_eval_obj = {}
  curr_eval_obj['global_step'] = global_step

  # Compute perplexity over the entire dataset.
  num_eval_batches = int(
      math.ceil(FLAGS.num_eval_examples / model.config.batch_size))

  start_time = time.time()
  img_sum_losses = 0.
  img_sum_weights = 0.
  text_sum_losses = 0.
  text_sum_weights = 0.
  img_sum_loss = 0.
  text_sum_loss = 0.
  for i in range(num_eval_batches):
    text_cross_entropy_losses, text_weights, image_losses, text_loss, image_loss = sess.run([
      model.text_cross_entropy_losses,
      model.text_cross_entropy_loss_weights,
      model.image_quadratic_losses,
      model.text_loss,
      model.image_loss
    ])
    text_sum_losses += np.sum(text_cross_entropy_losses * text_weights)
    text_sum_weights += np.sum(text_weights)
    
    img_sum_losses += np.sum(image_losses)
    img_sum_weights += image_losses.size
    
    img_sum_loss += image_loss
    text_sum_loss += text_loss
    
    if not i % 20: 
      tf.logging.info("Computed losses for %d of %d batches.", i + 1,
                      num_eval_batches)
  eval_time = time.time() - start_time
  
  tf.logging.info("Avg Text/Image loss = %f / %f (%.2g sec)", text_sum_loss/num_eval_batches, img_sum_loss/num_eval_batches, eval_time)

  perplexity_text = math.exp(text_sum_losses / text_sum_weights)
  tf.logging.info("Text perplexity = %f", perplexity_text)
  
  perplexity_img = math.exp(img_sum_losses / img_sum_weights)
  tf.logging.info("Image perplexity = %f", perplexity_img)
  
  curr_eval_obj['text_perplexity'] = perplexity_text
  curr_eval_obj['img_perplexity'] = perplexity_img
  curr_eval_obj['avg_text_loss'] = text_sum_loss/num_eval_batches
  curr_eval_obj['avg_img_loss'] = img_sum_loss/num_eval_batches

  # Log perplexity to the SummaryWriter.
  summary = tf.Summary()
  value = summary.value.add()
  value.simple_value = perplexity_text
  value.tag = "Text Perplexity"
  value = summary.value.add()
  value.simple_value = perplexity_img
  value.tag = "Image Perplexity"
  summary_writer.add_summary(summary, global_step)
  
  curr_eval_str = json.dumps(curr_eval_obj)
  dump_file.write(curr_eval_str + '\n')
  dump_file.flush()

  # Write the Events file to the eval directory.
  summary_writer.flush()
  tf.logging.info("Finished processing evaluation at global step %d.",
                  global_step)


def run_once(model, saver, summary_writer, summary_op, dump_file):
  """Evaluates the latest model checkpoint.

  Args:
    model: Instance of ShowAndTellModel; the model to evaluate.
    saver: Instance of tf.train.Saver for restoring model Variables.
    summary_writer: Instance of SummaryWriter.
    summary_op: Op for generating model summaries.
    dump_file: Reference to file, where evaluation results should be stored
  """
  print(FLAGS.checkpoint_dir)
  model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  if not model_path:
    tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                    FLAGS.checkpoint_dir)
    return

  with tf.Session() as sess:
    # Load model from checkpoint.
    tf.logging.info("Loading model from checkpoint: %s", model_path)
    saver.restore(sess, model_path)
    global_step = tf.train.global_step(sess, model.global_step.name)
    tf.logging.info("Successfully loaded %s at global step = %d.",
                    os.path.basename(model_path), global_step)
    if global_step < FLAGS.min_global_step:
      tf.logging.info("Skipping evaluation. Global step = %d < %d", global_step,
                      FLAGS.min_global_step)
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Run evaluation on the latest checkpoint.
    try:
      evaluate_model(
          sess=sess,
          model=model,
          global_step=global_step,
          summary_writer=summary_writer,
          summary_op=summary_op,
          dump_file=dump_file)
    except Exception as e:  # pylint: disable=broad-except
      tf.logging.error("Evaluation failed.")
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def run():
  """Runs evaluation in a loop, and logs summaries to TensorBoard."""
  # Create the evaluation directory if it doesn't exist.
  eval_dir = FLAGS.eval_dir
  if not tf.gfile.IsDirectory(eval_dir):
    tf.logging.info("Creating eval directory: %s", eval_dir)
    tf.gfile.MakeDirs(eval_dir)
    
  # generate eval dump file
  dump_file = open(os.path.join(eval_dir, 'evaluation.json'), 'a')

  g = tf.Graph()
  with g.as_default():
    # Build the model for evaluation.
    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern
    model = auto_encoder.AutoEncoder(model_config, mode="eval")
    model.build()

    # Create the Saver to restore model Variables.
    saver = tf.train.Saver()

    # Create the summary operation and the summary writer.
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(eval_dir)

    g.finalize()

    # Run a new evaluation run every eval_interval_secs.
    try:
      while True:
        start = time.time()
        tf.logging.info("Starting evaluation at " + time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()))
        run_once(model, saver, summary_writer, summary_op, dump_file)
        time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
        if time_to_next_eval > 0:
          time.sleep(time_to_next_eval)
    except KeyboardInterrupt:
      dump_file.close()


def main(unused_argv):
  assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
  assert FLAGS.eval_dir, "--eval_dir is required"
  run()


if __name__ == "__main__":
  tf.app.run()
