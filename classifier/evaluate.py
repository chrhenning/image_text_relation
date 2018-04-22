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
import os
import time

import json
import glob
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

import configuration
import classifier_model

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("eval_dir", "", "Directory to write event logs.")

tf.flags.DEFINE_integer("eval_interval_secs", 60,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples", 128,
                        "Number of examples for evaluation.")

tf.flags.DEFINE_integer("min_global_step", 50,
                        "Minimum global step to run evaluation.")

tf.logging.set_verbosity(tf.logging.INFO)

# Global variables to store the maximum score of an iteration reached so far and the global step where this score has been reached.
MAX_SCORE = -1
GS_MAX_SCORE = -1

def scoreIteration(jsonDump):
  """Helper function to evaluate the overall performance of a single evaluation.
  Currently, we evaluate the performance of an iteration based on the harmonic 
  mean of MI and SC accurcy. 

  Args:
    jsonDump: The jsonDump that is written to the json output file, containing 
      all the computed metrics for the currently considered global step 
      
  Output:
    The harmonic mean of MI and SC accuracy.
  """
  
  hm = lambda x,y: 2*x*y / (x+y)
  
  return hm(jsonDump['MI Accuracy'], jsonDump['SC Accuracy']) 

def backupCheckpoint(global_step):
  """Backup the checkpoint for global_step into an extra folder. This
  method is used to backup the checkpoint of the so far most promising 
  classifier model. 

  Args:
    global_step: Which checkpoint should be copied.
  """
  
  backup_dir = os.path.join(str(Path(FLAGS.eval_dir).parent), 'best')
  
  try:
    # delete previous backups
    if os.path.exists(backup_dir):
      shutil.rmtree(backup_dir)
    
    os.makedirs(backup_dir)
       
    for fn in glob.glob(os.path.join(FLAGS.checkpoint_dir, "model.ckpt-"+str(global_step)+".*")):
      shutil.copy(fn, backup_dir)
  
  except Exception as e:
    print("Could not backup the checkpoint at global step %d successfully" % (global_step))
    print(e)
    # delete already copied part
    if os.path.exists(backup_dir):
      shutil.rmtree(backup_dir)

def computeRMS(y_true, y_pred, distances):
  """Helper function to compute RMS of a given multiclass problem.

  Args:
    y_true: The ground truth labels
    y_pred: The predicted labels
    distances: A symmetric matrix defining the distance between two labels.
  """
  
  rms = 0.0;
  
  for i in range(len(y_true)):
    yt = y_true[i]
    yp = y_pred[i]
    rms += distances[yt][yp]**2
  
  rms = math.sqrt(rms/len(y_true))
  
  return rms  

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
  mi_loss_sum = 0.
  sc_loss_sum = 0.
  
  mi_pred = []
  mi_true = []
  
  sc_pred = []
  sc_true = []
  
  for i in range(num_eval_batches):
    mi_labels, sc_labels, mi_loss, sc_loss, mi_logits, sc_logits = sess.run([
      model.mi_labels,
      model.sc_labels,
      model.mi_loss,
      model.sc_loss,
      model.mi_logits,
      model.sc_logits
    ])
    
    mi_loss_sum += mi_loss
    sc_loss_sum += sc_loss
    
    if model.config.mi_is_multiclass_problem:
      mi_pred += mi_logits.argmax(axis=1).tolist()
    else:
      mi_pred += np.squeeze(mi_logits).tolist()
    
    mi_true += mi_labels.tolist()    
    
    if model.config.sc_is_multiclass_problem:
      sc_pred += ((sc_logits.argmax(axis=1) - 2)/2.0).tolist() 
    else:
      sc_pred += np.squeeze(sc_logits).tolist()

    sc_true += sc_labels.tolist()
    
    if not i % 20: 
      tf.logging.info("Computed losses for %d of %d batches.", i + 1,
                      num_eval_batches)
  eval_time = time.time() - start_time

  tf.logging.info("Avg MI/SC loss = %f / %f (%.2g sec)", mi_loss_sum/num_eval_batches, sc_loss_sum/num_eval_batches, eval_time)
  
  # Log metrics to the SummaryWriter.
  summary = tf.Summary()
  def addValueToSummary(tag, val):
    value = summary.value.add()
    value.simple_value = val
    value.tag = tag
    curr_eval_obj[tag] = val
    
  # TODO We do the same for both measurements (MI and SC), so put the code in an extra function
  
  # compute metrics for semantic correlation
  labels = range(model.config.num_mi_labels)
  ilabels = labels
  imi_true = [int(l) for l in mi_true]
  if model.config.mi_is_multiclass_problem:
    imi_pred = [int(l) for l in mi_pred]
  else:
    # compute multiclass metrics by assuming the closest label as the predicted one
    imi_pred = [int(i) for i in (np.rint(np.array(mi_pred)))]
    # Crop to correct range
    max_l = model.config.num_mi_labels - 1
    imi_pred = [max_l if l > max_l else 0 if l < 0 else l for l in imi_pred]
  
  # compute metrics for mutual information
  acc = accuracy_score(imi_true, imi_pred)
  addValueToSummary("MI Accuracy", acc)
  
  prec, rec, f1, _ = precision_recall_fscore_support(imi_true, imi_pred, labels=ilabels, average='macro')
  tf.logging.info("Mutual Information - Macro F1 = %f", f1)
  addValueToSummary("MI Macro Precision", prec)
  addValueToSummary("MI Macro Recall", rec)
  addValueToSummary("MI Macro F1-Score", f1)
  
  prec, rec, f1, _ = precision_recall_fscore_support(imi_true, imi_pred, labels=ilabels, average='micro')
  tf.logging.info("Mutual Information - Micro F1 = %f", f1)
  addValueToSummary("MI Micro Precision", prec)
  addValueToSummary("MI Micro Recall", rec)
  addValueToSummary("MI Micro F1-Score", f1)
  
  prec, rec, f1, _ = precision_recall_fscore_support(imi_true, imi_pred, labels=ilabels, average='weighted')
  tf.logging.info("Mutual Information - Weighted F1 = %f", f1)
  addValueToSummary("MI Weighted Precision", prec)
  addValueToSummary("MI Weighted Recall", rec)
  addValueToSummary("MI Weighted F1-Score", f1)
  
  prec, rec, f1, supp = precision_recall_fscore_support(imi_true, imi_pred, labels=ilabels, average=None)
  for i in range(len(prec)):
    label = labels[i]
    addValueToSummary("MI Precision of Label %d" % (label), prec[i])
    addValueToSummary("MI Recall of Label %d" % (label), rec[i])
    addValueToSummary("MI F1-Score of Label %d" % (label), f1[i])
    addValueToSummary("MI Support of Label %d" % (label), int(supp[i]))

  if model.config.mi_is_multiclass_problem:
    rms = computeRMS(imi_true, imi_pred, model.config.mi_label_distances)
    tf.logging.info("Mutual Information - RMS with predefined distances = %f", rms)
    addValueToSummary("MI Distance Aware RMS", rms)
  
  else:
    # considered as regression problem
    quadratic_distance = np.power(np.subtract(mi_true, mi_pred),2)
    addValueToSummary("MI Min Quadratic Distance",  quadratic_distance.min())
    addValueToSummary("MI Mean Quadratic Distance",  quadratic_distance.mean())
    addValueToSummary("MI Max Quadratic Distance",  quadratic_distance.max())
    
    # compute RMS. Values are cropped to range [0,num_mi_labels]. Distances smaller 0.5 are considered as 0.
    rms = 0.0;  
    for i in range(len(mi_true)):
      yt = mi_true[i]
      yp = mi_pred[i]
      # crop to range
      yp = model.config.num_mi_labels if yp > model.config.num_mi_labels else 0 if yp < 0 else yp
      distance = math.fabs(yp - yt)
      # if the distance is too small, then we consider it as zero anyway
      distance = 0 if distance < 1 else distance - 1
      rms += distance**2
    rms = math.sqrt(rms/len(mi_true))
    tf.logging.info("Mutual Information - RMS with cropped distances = %f", rms)
    addValueToSummary("MI Distance Aware RMS", rms) 

  # compute metrics for semantic correlation
  labels = [-1.0, -0.5, 0.0, 0.5, 1.0]
  ilabels = [int(l * 2 + 2) for l in labels]
  isc_true = [int(l * 2 + 2) for l in sc_true]
  if model.config.sc_is_multiclass_problem:
    isc_pred = [int(l  * 2 + 2) for l in sc_pred]
  else:
    # compute multiclass metrics by assuming the closest label as the predicted one
    isc_pred = [int(i) for i in (np.rint(2*np.array(sc_pred)) + 2)]
    # Crop to correct range
    max_l = model.config.num_sc_labels - 1 
    isc_pred = [max_l if l > max_l else 0 if l < 0 else l for l in isc_pred]
    
  acc = accuracy_score(isc_true, isc_pred)
  addValueToSummary("SC Accuracy", acc)

  prec, rec, f1, _ = precision_recall_fscore_support(isc_true, isc_pred, labels=ilabels, average='macro')
  tf.logging.info("Semantic Correlation - Macro F1 = %f", f1)
  addValueToSummary("SC Macro Precision", prec)
  addValueToSummary("SC Macro Recall", rec)
  addValueToSummary("SC Macro F1-Score", f1)
  
  prec, rec, f1, _ = precision_recall_fscore_support(isc_true, isc_pred, labels=ilabels, average='micro')
  tf.logging.info("Semantic Correlation - Micro F1 = %f", f1)
  addValueToSummary("SC Micro Precision", prec)
  addValueToSummary("SC Micro Recall", rec)
  addValueToSummary("SC Micro F1-Score", f1)
  
  prec, rec, f1, _ = precision_recall_fscore_support(isc_true, isc_pred, labels=ilabels, average='weighted')
  tf.logging.info("Semantic Correlation - Weighted F1 = %f", f1)
  addValueToSummary("SC Weighted Precision", prec)
  addValueToSummary("SC Weighted Recall", rec)
  addValueToSummary("SC Weighted F1-Score", f1)
  
  prec, rec, f1, supp = precision_recall_fscore_support(isc_true, isc_pred, labels=ilabels, average=None)
  for i in range(len(prec)):
    label = labels[i]
    addValueToSummary("SC Precision of Label %.1f" % (label), prec[i])
    addValueToSummary("SC Recall of Label %.1f" % (label), rec[i])
    addValueToSummary("SC F1-Score of Label %.1f" % (label), f1[i])
    addValueToSummary("SC Support of Label %.1f" % (label), int(supp[i]))

  if model.config.sc_is_multiclass_problem:
    rms = computeRMS(isc_true, isc_pred, model.config.sc_label_distances)
    tf.logging.info("Semantic Correlation - RMS with predefined distances = %f", rms)
    addValueToSummary("SC Distance Aware RMS", rms)
  
  else:
    # considered as regression problem
    quadratic_distance = np.power(np.subtract(sc_true, sc_pred),2)
    addValueToSummary("SC Min Quadratic Distance",  quadratic_distance.min())
    addValueToSummary("SC Mean Quadratic Distance",  quadratic_distance.mean())
    addValueToSummary("SC Max Quadratic Distance",  quadratic_distance.max())
    
    # compute RMS. Values are cropped to range [-1,1]. Distances smaller 0.5 are considered as 0.
    rms = 0.0;  
    for i in range(len(sc_true)):
      yt = sc_true[i]
      yp = sc_pred[i]
      # crop to range
      yp = 1 if yp > 1 else -1 if yp < -1 else yp
      distance = math.fabs(yp - yt)
      distance = 0 if distance < 0.5 else distance - 0.5
      rms += distance**2
    rms = math.sqrt(rms/len(sc_true))
    tf.logging.info("Semantic Correlation - RMS with cropped distances = %f", rms)
    addValueToSummary("SC Distance Aware RMS", rms) 

  summary_writer.add_summary(summary, global_step)
  # Write the Events file to the eval directory.
  summary_writer.flush()
  
  curr_eval_str = json.dumps(curr_eval_obj)
  dump_file.write(curr_eval_str + '\n')
  dump_file.flush()
  
  # check if this is the best performing iteration so far
  global MAX_SCORE, GS_MAX_SCORE
  score = scoreIteration(curr_eval_obj)
  if score >= MAX_SCORE:
    MAX_SCORE = score
    GS_MAX_SCORE = curr_eval_obj['global_step']
    backupCheckpoint(GS_MAX_SCORE)
  
  tf.logging.info("Finished processing evaluation at global step %d.",
                  global_step)


# We do not want to run the evaluation of the same checkpoint multiple times.
# So we store the last evaluated global step
RECENT_GS = -1

def run_once(model, saver, summary_writer, summary_op, dump_file):
  """Evaluates the latest model checkpoint.

  Args:
    model: Instance of ShowAndTellModel; the model to evaluate.
    saver: Instance of tf.train.Saver for restoring model Variables.
    summary_writer: Instance of SummaryWriter.
    summary_op: Op for generating model summaries.
    dump_file: Reference to file, where evaluation results should be stored
  """
  global RECENT_GS
  
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
      
    # make sure we do not evaluate something, that we already evaluated
    if RECENT_GS != -1 and RECENT_GS == global_step:
      return
    RECENT_GS = global_step

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

  # We are going to dump all evaluation metrics at each step into a file to have
  # it easily accessible for later evaluation.
  dump_fn = os.path.join(eval_dir, 'evaluation.json')
  
  # Check if the dumpfile already exists
  # If so, we initialize the global variables MAX_SCORE and GS_MAX_SCORE at first
  if os.path.exists(dump_fn):
    global MAX_SCORE, GS_MAX_SCORE
    
    with open(dump_fn, 'r') as f:
      for line in f:
        curr = json.loads(line)
        score = scoreIteration(curr)
        if score >= MAX_SCORE:
          MAX_SCORE = score
          GS_MAX_SCORE = curr['global_step']

  # generate eval dump file
  dump_file = open(dump_fn, 'a')
    
  g = tf.Graph()
  with g.as_default():
    # Build the model for evaluation.
    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern
    model = classifier_model.Classifier(model_config, mode="eval")
    model.build()

    # Create the Saver to restore model Variables.
    saver = tf.train.Saver()

    # Create the summary operation and the summary writer.
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(eval_dir)

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
