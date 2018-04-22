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

"""Converts MSCOCO, Wiki and BBC image-text pairs to TFRecord file format with SequenceExample protos.

The mscoco dataset is expected to reside in coco_sample_file, which is a jsonl file,
that contains the absolute paths to the images.

The simple wiki dataset is expected to reside in wiki_sample_file, which is a jsonl file,
that contains the relative paths to the images.

The BBC dataset is expected to reside in bbc_sample_file, which is a jsonl file,
that contains the relative paths to the images.

This script converts the combined datasets into sharded data files consisting
of 16 and 2 TFRecord files, respectively:

  output_dir/train-00000-of-00016
  output_dir/train-00001-of-00016
  ...
  output_dir/train-00015-of-00016

and

  output_dir/test-00000-of-00002
  ...
  output_dir/test-00001-of-00002

Each TFRecord file contains ~35 records. Each record within the TFRecord file
is a serialized SequenceExample proto consisting of precisely one image-text
pair.

The SequenceExample proto contains the following fields:

  context:
    image/image_id: integer MSCOCO image identifier or -1
    image/data: string containing JPEG encoded image in RGB colorspace
    annotation/mi: Integer representing the mutual information label
    annotation/sc: Float representing the semantic correlation label

  feature_lists:
    image/sentences: list of strings containing the (tokenized) text (concatenated sentences) words
    image/sentences_splits: list of integers corresponding to the start index of each sentence in image/sentences
    image/sentences_ids: list of arrays of integer ids corresponding to the text words
    
Note, that all sentences are truncated/padded to the length of max_num_tokens.
So, the feature list 'sentences_ids', is a sequence of fixed size feature vectors.
In order to avoid huge memory usage for outlier texts (texts with a huge amount of sentences cause the whole batch  
to take this sequence size and load a word embedding for each token in this sequences of token lists). Therefore,
too long texts are cropped.

The sentences are tokenized using the NLTK (http://www.nltk.org/) word tokenizer.
The vocabulary of word identifiers is the same as used by the autoencoder. Only tokens appearing
at least 10 times are considered; all other words get the "unknown" word id.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
import json
import os.path
import random
import sys
import threading



import nltk.tokenize
import numpy as np
import tensorflow as tf

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#from nltk import stem
#porter = stem.porter.PorterStemmer()

tf.flags.DEFINE_string("coco_sample_file", "coco-anno-samples.jsonl",
                       "File containing the samples from the mscoco dataset (Samples must contain absolute image filename).")
                       
tf.flags.DEFINE_string("wiki_sample_file", "wiki-anno-samples.jsonl",
                       "The file that contains all the information about the wiki dataset. (contains relatice image paths)")
                       
tf.flags.DEFINE_string("bbc_sample_file", "bbc-anno-samples.jsonl",
                       "The file that contains all the information about the bbc dataset. (contains relatice image paths)")

tf.flags.DEFINE_string("output_dir", "/tmp/", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 16,
                        "Number of shards in training TFRecord files.")
#tf.flags.DEFINE_integer("val_shards", 16,
#                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 2,
                        "Number of shards in testing TFRecord files.")

#tf.flags.DEFINE_string("start_sentence", "<S>",
#                       "Special sentence added to the beginning of each text.")
#tf.flags.DEFINE_string("end_sentence", "</S>",
#                       "Special sentence added to the end of each text.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_string("word_counts_input_file", "/tmp/word_counts.txt",
                       "Input vocabulary (Generated by the autoencoder).")
tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")
tf.flags.DEFINE_integer("max_num_tokens", 40,
                        "Maximum number of tokens in a sentence. Must be the same as used when training the autoencoder.") 
tf.flags.DEFINE_integer("max_num_sentences", 50,
                        "Maximum number of sentences in a text. ")
                        
tf.flags.DEFINE_string("br_am_dict", "br_am_dict.txt",
                       "A dictionary containing translations from british english to american english")     
                       
tf.flags.DEFINE_integer("lemmatize_tokens", 0,
                        "Set to 1, if tokens should be lemmatized")  

tf.flags.DEFINE_boolean("only_photographs", False,
                        "Consider only the image type 'photographs' to generate samples.")  

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "sentences", "sc", "mi"])

# dictionary is read from file and stored in this variable
br_am_dict = None

class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self, vocab, unk_id):
    """Initializes the vocabulary.

    Args:
      vocab: A dictionary of word to word_id.
      unk_id: Id of the special 'unknown' word.
    """
    self._vocab = vocab
    self._unk_id = unk_id

  def word_to_id(self, word):
    """Returns the integer id of a word string."""
    if word in self._vocab:
      return self._vocab[word]
    else:
      return self._unk_id


class ImageDecoder(object):
  """Helper class for decoding images in TensorFlow."""

  def __init__(self):
    # Create a single TensorFlow Session for all image decoding calls.
    self._sess = tf.Session()

    # TensorFlow ops for JPEG decoding.
    self._encoded_jpeg = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

  def decode_jpeg(self, encoded_jpeg):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._encoded_jpeg: encoded_jpeg})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _float_feature(value):
  """Wrapper for inserting a float Feature into a SequenceExample proto."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_features(values):
  """Wrapper for inserting int64 Features into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v.encode('utf8')) for v in values])
  
def _int64_feature_list_of_lists(values):
  """Wrapper for inserting an int64 FeatureList of arrays into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_features(v) for v in values])


def _to_sequence_example(image, decoder, vocab):
  """Builds a SequenceExample proto for an image-text pair.

  Args:
    image: An ImageMetadata object.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.

  Returns:
    A SequenceExample proto.
  """
  with tf.gfile.FastGFile(image.filename.encode('utf8', 'surrogateescape'), "r") as f:
    encoded_image = f.read()

  try:
    decoder.decode_jpeg(encoded_image)
  except (tf.errors.InvalidArgumentError, AssertionError):
    print("Skipping file with invalid JPEG data: %s" % image.filename)
    return

  context = tf.train.Features(feature={
      "image/image_id": _int64_feature(image.image_id),
      "image/data": _bytes_feature(encoded_image),
      "annotation/mi": _int64_feature(image.mi),
      "annotation/sc": _float_feature(image.sc)
  })

  # concat all sentences
  sentences = []
  # in order to be able to later extract the single sentences from the concatenated list, we store the start index of each sentence
  sentences_splits = []
  # contains an array of max_num_tokens integers for each sentence
  sentences_ids = []
  # truncate sentences to avoid too long texts
  for sentence in image.sentences:
    sentence_ids = [vocab.word_to_id(word) for word in sentence]
    # pad sentence_ids if necessary
    sentence_ids += [0] * (FLAGS.max_num_tokens - len(sentence_ids))
    
    sentences_splits.append(len(sentences))
    sentences += sentence
    sentences_ids.append(sentence_ids)
    
  feature_lists = tf.train.FeatureLists(feature_list={
      "image/sentences": _bytes_feature_list(sentences),
      "image/sentences_splits": _int64_feature_list(sentences_splits),
      "image/sentences_ids": _int64_feature_list_of_lists(sentences_ids)
  })
  sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)

  return sequence_example


def _process_image_files(thread_index, ranges, name, images, decoder, vocab,
                         num_shards):
  """Processes and saves a subset of images as TFRecord files in one thread.

  Args:
    thread_index: Integer thread identifier within [0, len(ranges)].
    ranges: A list of pairs of integers specifying the ranges of the dataset to
      process in parallel.
    name: Unique identifier specifying the dataset.
    images: List of ImageMetadata.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
  """
  # Each thread produces N shards where N = num_shards / num_threads. For
  # instance, if num_shards = 128, and num_threads = 2, then the first thread
  # would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  # e.g. assume there are 16 images in this batch and the batch should be split into
  # 2 shards. Then if the considered range is [0,16], we expect as output: [0,8,16]
  shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
  
  #print(str(thread_index) + ' ranges: ' + str(ranges))
  #print(str(thread_index) + ' shard_ranges: ' + str(shard_ranges))

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    #print(str(thread_index) + ' images_in_shard: ' + str(len(images_in_shard)))
    for i in images_in_shard:
      image = images[i]

      sequence_example = _to_sequence_example(image, decoder, vocab)
      if sequence_example is not None:
        writer.write(sequence_example.SerializeToString())
        shard_counter += 1
        counter += 1

      if not counter % 1000:
        print("%s [thread %d]: Processed %d of %d items in thread batch." %
              (datetime.now(), thread_index, counter, num_images_in_thread))
        sys.stdout.flush()

    writer.close()
    print("%s [thread %d]: Wrote %d image-text pairs to %s" %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print("%s [thread %d]: Wrote %d image-text pairs to %d shards." %
        (datetime.now(), thread_index, counter, num_shards_per_batch))
  sys.stdout.flush()


def _process_dataset(name, images, vocab, num_shards):
  """Processes a complete data set and saves it as a TFRecord.

  Args:
    name: Unique identifier specifying the dataset.
    images: List of ImageMetadata.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
  """

  # Break the images into num_threads batches. Batch i is defined as
  # images[ranges[i][0]:ranges[i][1]].
  num_threads = min(num_shards, FLAGS.num_threads)
  spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a utility for decoding JPEG images to run sanity checks.
  decoder = ImageDecoder()

  # Launch a thread for each batch.
  print("Launching %d threads for spacings: %s" % (num_threads, ranges))
  for thread_index in range(len(ranges)):
    args = (thread_index, ranges, name, images, decoder, vocab, num_shards)
    t = threading.Thread(target=_process_image_files, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print("%s: Finished processing all %d image-text pairs in data set '%s'." %
        (datetime.now(), len(images), name))


def _read_vocab(vocab_file):
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
  unk_id = len(reverse_vocab) + 1
  vocab_dict = dict([(x, y+1) for (y, x) in enumerate(reverse_vocab)])
  vocab = Vocabulary(vocab_dict, unk_id)

  return vocab


def _read_dictionary():
  """Read the british-american dictionary
  into an internal representation.
  """
  global br_am_dict
  
  br_am_dict = dict()
  
  with open(FLAGS.br_am_dict, 'r') as f:
    for line in f:
      if line.startswith('//'): # ignore comment lines
        continue
        
      line = line.strip()
      tokens = line.split()
      if len(tokens) != 2: # should only happen for one strange word
        continue
      
      br_word, am_word = tokens
      br_am_dict[br_word] = am_word

num_translations = 0

def _process_sentence(sentence, max_num_tokens, translate=False):
  """Processes a sentence string into a list of tonenized words.

  Args:
    sentence: A string sentence.
    max_num_tokens: Maximum number of tokens in a sentence
    translate: Whether the sentence should be translated from british to american english.

  Returns:
    A list of strings; the tokenized sentence.
  """
  tokenized_sentence = []
  tokenized_sentence.extend(nltk.tokenize.word_tokenize(sentence.lower()))
  # truncate sentences
  tokenized_sentence = tokenized_sentence[:max_num_tokens]
  
  # lemmatize tokens
  if FLAGS.lemmatize_tokens:
    tokenized_sentence = [lemmatizer.lemmatize(t) for t in tokenized_sentence]
    
  # translate tokens from british to american english
  if translate:
    global num_translations, br_am_dict
    
    if br_am_dict is None:
      _read_dictionary()
      
    tmp = []
    for t in tokenized_sentence:
      if t in br_am_dict:
        tmp.append(br_am_dict[t])
        #print('%s has been translated to %s' % (t, tmp[-1]))
        num_translations += 1
      else:
        tmp.append(t)
    tokenized_sentence = tmp
  
  return tokenized_sentence
  
  
def _process_text(text):
  """Processes a list of sentences to add special start and end sequence
    specifier and to truncate text if necessary.

  Args:
    Text: A list of list of tokens.

  Returns:
    A list of list of tokens.
  """
  # truncate sentences, if necessary
  text = text[:FLAGS.max_num_sentences]
  
  # truncate sentences, if necessary
  #text = text[:FLAGS.max_num_sentences-2]
  # start and end sentence are zero padded later on
  #text = [[FLAGS.start_sentence]] + text
  #text += [[FLAGS.end_sentence]]

  return text

def _load_and_process_metadata_coco(coco_file):
  """Loads image metadata from a JSON Line file containing the mscoco
     dataset with fake annotations

  Args:
    wiki_file: JSON Line file containing the image-text pairs.

  Returns:
    A list of ImageMetadata.
  """
  image_metadata = []
  print("Proccessing mscoco dataset.")

  num_sentences = 0
  
  with tf.gfile.FastGFile(coco_file, "r") as f:
    for pair in f:
      sample = json.loads(pair)
 
      filename = sample['imgpath']
      #assert os.path.isfile(filename)
      # just concatenate all sentences
      # TODO distinguish between titles and normal text (maybe by different enclosing words)
      # just as we humans do
      # The text field is a list of tuples (title, paragraph), which themselves are lists of sentences
      sentences = [_process_sentence(s, FLAGS.max_num_tokens) for s in sample['text']]
      image_metadata.append(ImageMetadata(-1, filename, _process_text(sentences), sample['annotation']['sc'], sample['annotation']['mi'])) # FIXME image_id -1
      num_sentences += len(sentences)

  print("Finished processing %d sentences for %d images in %s" %
        (num_sentences, len(image_metadata), coco_file))

  return image_metadata

def _load_and_process_metadata_wiki(wiki_file):
  """Loads image metadata from a JSON Line file containing the simple-wiki
     dataset with annotations and processes the texts.

  Args:
    wiki_file: JSON Line file containing the image-text pairs.

  Returns:
    A list of ImageMetadata.
  """
  image_metadata = []
  print("Proccessing simple wiki dataset.")
  
  image_base_dir = os.path.dirname(wiki_file)
  num_sentences = 0
  
  with tf.gfile.FastGFile(wiki_file, "r") as f:
    for pair in f:
      sample = json.loads(pair)
 
      filename = os.path.join(image_base_dir, sample['imgpath'])
      #assert os.path.isfile(filename)
      # just concatenate all sentences
      # TODO distinguish between titles and normal text (maybe by different enclosing words)
      # just as we humans do
      # The text field is a list of tuples (title, paragraph), which themselves are lists of sentences
      
      # if only samples with image type 'photograp'h should be considered
      if FLAGS.only_photographs:
        if not sample['annotation']['type'] == 'Photograph':
          continue
      
      sentences = [_process_sentence(s, FLAGS.max_num_tokens) for tup in sample['text'] for item in tup for s in item]
      image_metadata.append(ImageMetadata(-1, filename, _process_text(sentences), sample['annotation']['sc'], sample['annotation']['mi'])) # FIXME image_id -1
      num_sentences += len(sentences)

  print("Finished processing %d sentences for %d images in %s" %
        (num_sentences, len(image_metadata), wiki_file))

  return image_metadata

def _load_and_process_metadata_bbc(bbc_file):
  """Loads image metadata from a JSON Line file containing the BBC news
     dataset with annotations and processes the texts.

  Args:
    bbc_file: JSON Line file containing the image-text pairs.

  Returns:
    A list of ImageMetadata.
  """
  image_metadata = []
  print("Proccessing BBC dataset.")
  
  image_base_dir = os.path.dirname(bbc_file)
  num_sentences = 0
  
  with tf.gfile.FastGFile(bbc_file, "r") as f:
    for pair in f:
      sample = json.loads(pair)
 
      filename = os.path.join(image_base_dir, sample['image'])
      #assert os.path.isfile(filename)
      # just concatenate all sentences
      # TODO distinguish between titles and normal text (maybe by different enclosing words)
      # just as we humans do
      # The text field is a list of tuples (title, paragraph), which themselves are lists of sentences
      
      # if only samples with image type 'photograp'h should be considered
      if FLAGS.only_photographs:
        if not sample['annotation']['type'] == 'Photograph':
          continue
          
      sentences = [_process_sentence(s, FLAGS.max_num_tokens, translate=True) for tup in sample['text'] for item in tup for s in item]
      image_metadata.append(ImageMetadata(-1, filename, _process_text(sentences), sample['annotation']['sc'], sample['annotation']['mi'])) # FIXME image_id -1
      num_sentences += len(sentences)

  print("Finished processing %d sentences for %d images in %s" %
        (num_sentences, len(image_metadata), bbc_file))

  return image_metadata
  
def _print_label_distribution(dataset):
  """Prints the label distribution in a given dataset

  Args:
    dataset: List of ImageMetadata.
  """
  mi_labels = {}
  sc_labels = {}
  
  for sample in dataset:
    mi_labels.setdefault(sample.mi, 0)
    sc_labels.setdefault(sample.sc, 0)
    
    mi_labels[sample.mi] += 1
    sc_labels[sample.sc] += 1
  
  mi_labels = mi_labels.items()
  sc_labels = sc_labels.items()
  
  mi_labels = sorted(mi_labels, key=lambda x: float(x[0]))
  sc_labels = sorted(sc_labels, key=lambda x: float(x[0]))
  
  def printTable(tuples):
    prependingSpaces = 3
    heading = ['Label'] + [x[0] for x in tuples]
    row_format = ' ' * prependingSpaces + "# {:<15} | " + "{:>15}" * (len(heading) - 1)
  
    print(row_format.format(*heading))
    print(' ' * prependingSpaces + "# " + "-"*15  +"-|-" + "-" * 15 * (len(heading) - 1))
    print(row_format.format('Count', *[x[1] for x in tuples]))
    
  print("   MI label distribution:")
  printTable(mi_labels)
  print()
  print("   SC label distribution:")
  printTable(sc_labels)
  print()

def main(unused_argv):
  def _is_valid_num_shards(num_shards):
    """Returns True if num_shards is compatible with FLAGS.num_threads."""
    return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

  assert _is_valid_num_shards(FLAGS.train_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
  #assert _is_valid_num_shards(FLAGS.val_shards), (
  #    "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
  assert _is_valid_num_shards(FLAGS.test_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  
  # Load image metadata from caption files.
  mscoco_dataset = _load_and_process_metadata_coco(FLAGS.coco_sample_file)

  # load image metadata and texts from wiki dataset
  wiki_dataset = _load_and_process_metadata_wiki(FLAGS.wiki_sample_file)
  
  # load image metadata and texts from BBC news dataset
  bbc_dataset = _load_and_process_metadata_bbc(FLAGS.bbc_sample_file)
  
  global num_translations
  print("%d words have been translated from british to american english" % (num_translations))
  
  # Redistribute the data as follows:
  #   train_dataset = 89% of dataset.
  #   val_dataset = 0% of dataset.
  #   test_dataset = 11% of dataset.
  
  dataset = mscoco_dataset + wiki_dataset + bbc_dataset
  
  # Shuffle the ordering of images. Make the randomization repeatable.
  random.seed(12345)
  random.shuffle(dataset)
  
  train_dataset = []
  test_dataset = []
  
  # stratisfied sampling 
  # build groups of samples according to MI
  mi_groups = {}
  for sample in dataset:
    mi_groups.setdefault(sample.mi, [])
    mi_groups[sample.mi].append(sample)
    
  # build groups of each mi_group according to SC
  sc_groups = {}
  for mi in mi_groups.keys():
    sc_groups[mi] = {}
    for sample in mi_groups[mi]:
      sc_groups[mi].setdefault(sample.sc, [])
      sc_groups[mi][sample.sc].append(sample)
      
  # Now we have to distribute the grouped samples equally among the shards
  for mi in sc_groups.keys():
    mi_group = sc_groups[mi]
    for sc in mi_group.keys():
      train_cutoff = round(0.89 * len(mi_group[sc]))
      train_dataset += mi_group[sc][:train_cutoff]
      test_dataset += mi_group[sc][train_cutoff:]  
      
  # their should be a good mixing among shards (however, we can make the queue of shards big enough, such that it doen't matter)
  random.shuffle(train_dataset)    
  random.shuffle(test_dataset)  
  
  print('The data contains %d samples all together.' % (len(train_dataset) + len(test_dataset)))
  print('Sample distribution: Training (%d), Test (%d)' % (len(train_dataset), len(test_dataset)))

  print('The label distribution in the training dataset:')
  _print_label_distribution(train_dataset)
  print()
  print('The label distribution in the test dataset:')
  _print_label_distribution(test_dataset)
  print()
  print('Overall label distribution:')
  _print_label_distribution(train_dataset+test_dataset)
  print()

  # Read vocabulary from the dumped autoencoder vocab.
  vocab = _read_vocab(FLAGS.word_counts_input_file)

  _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)
  #_process_dataset("val", val_dataset, vocab, FLAGS.val_shards)
  _process_dataset("test", test_dataset, vocab, FLAGS.test_shards)


if __name__ == "__main__":
  tf.app.run()
