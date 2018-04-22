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
#
# Part of the code in this file i taken from here:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

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
import word2vec

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("vocab_file", "",
                       "File containing vobaulary.")
tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("eval_dir", "", "Directory to write event logs.")

tf.flags.DEFINE_integer("eval_interval_secs", 60,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples", 1024,
                        "Number of examples for evaluation.")

tf.flags.DEFINE_integer("min_global_step", 5000,
                        "Minimum global step to run evaluation.")

tf.logging.set_verbosity(tf.logging.INFO)

def _read_vocab(vocab_file=FLAGS.vocab_file):
  """Reads the vocabulary of word to word_id.

  The vocabulary is read from disk from a text file of word counts. The id of each
  word in the file is its corresponding 1-based line number (thus, first line has ID 1).

  Args:
    vocab_file: File containing the vocabulary.

  Returns:
    A Vocabulary object.
  """
  print("Reading vocabulary.")

  word_counts = []

  # Write out the word counts file.
  with tf.gfile.FastGFile(vocab_file, "r") as f:
    for line in f:
      word, count = line.strip().split(' ')
      word_counts.append((word,int(count)))
  print("Words in vocabulary:", len(word_counts))

  # Create the vocabulary dictionary.
  # Make sure, that the ID 0 (padding value) is not used for the vocabulary
  reverse_vocab = [x[0] for x in word_counts]
  reverse_vocab.append("UNK")
  
  unk_id = len(reverse_vocab) + 1
  
  vocab_dict = dict([(x, y+1) for (y, x) in enumerate(reverse_vocab)])

  return vocab_dict, reverse_vocab


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
  #summary_str = sess.run(summary_op)
  #summary_writer.add_summary(summary_str, global_step)
  
  # Log evaluation into this dict
  curr_eval_obj = {}
  curr_eval_obj['global_step'] = global_step
  
  _, reverse_dictionary = _read_vocab()
    
  # We pick a random validation set to sample nearest neighbors. Here we limit the
  # validation samples to the words that have a low numeric ID, which by
  # construction are also the most frequent.
  valid_size = 16     # Random set of words to evaluate similarity on.
  valid_window = 100  # Only pick dev samples in the head of the distribution.
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)

  sim, final_embeddings = sess.run([model.similarity, model.normalized_embeddings], feed_dict = { "valid_dataset:0": valid_examples })
  for i in range(valid_size):
    valid_word = reverse_dictionary[valid_examples[i]]
    top_k = 8  # number of nearest neighbors
    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
    log_str = "Nearest to %s:" % valid_word
    for k in range(top_k):
      close_word = reverse_dictionary[min(nearest[k], len(reverse_dictionary)-1)]
      log_str = "%s %s," % (log_str, close_word)
    print(log_str)

  # Visualize the embeddings.
  def plot_with_labels(low_dim_embs, labels, filename=os.path.join(FLAGS.eval_dir,'tsne.png')):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
      x, y = low_dim_embs[i, :]
      plt.scatter(x, y)
      plt.annotate(label,
                   xy=(x, y),
                   xytext=(5, 2),
                   textcoords='offset points',
                   ha='right',
                   va='bottom')

    plt.savefig(filename)

  try:
    from sklearn.manifold import TSNE
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)

  except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
  
  curr_eval_str = json.dumps(curr_eval_obj)
  #dump_file.write(curr_eval_str + '\n')

  # Write the Events file to the eval directory.
  #summary_writer.flush()
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

    # Run evaluation on the latest checkpoint.
    evaluate_model(
        sess=sess,
        model=model,
        global_step=global_step,
        summary_writer=summary_writer,
        summary_op=summary_op,
        dump_file=dump_file)


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
    model = word2vec.Word2Vec(model_config, mode="eval")
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
  assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
  assert FLAGS.eval_dir, "--eval_dir is required"
  run()


if __name__ == "__main__":
  tf.app.run()
