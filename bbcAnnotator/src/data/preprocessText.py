#!/usr/bin/env python3
# This tool will preprocess the data of the bbc dataset
# Preprocessing here means simply to bring it in an appropriate form that allows easy access to image, text pairs
# and to split the corresponding text already into sentences

import argparse
import os
import nltk
import json
import sys
import traceback
from lxml import etree

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
htmlParser = etree.HTMLParser()

global OUTPUT_FILE

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--train', type=str, help="Folder, containing training data", default='./data_train')
  parser.add_argument('--test', type=str, help="Folder, containing test data", default='./data_test')
  parser.add_argument('--outname', type=str, help="The name of the jsonl file, that is generated by this script.", default='bbc-samples.jsonl')
  args = parser.parse_args()

  train_files_path = args.train
  test_files_path = args.test
  outname = os.path.join(args.outname)

  if not os.path.isdir(train_files_path):
    raise(Exception('Directory ' + train_files_path + ' does not exist.'))
    
  if not os.path.isdir(test_files_path):
    raise(Exception('Directory ' + test_files_path + ' does not exist.'))
    
  OUTPUT_FILE = open(outname, 'w')
  
  train_files = [f for f in os.listdir(train_files_path) if os.path.isfile(os.path.join(train_files_path, f))]
  test_files = [f for f in os.listdir(test_files_path) if os.path.isfile(os.path.join(test_files_path, f))]

  # dictionary that contains filename as key and maps onto all three files plus meta data
  files = {}

  def insert_filenames(filenames, train, path):
    for filename in filenames:
      fn, fe = os.path.splitext(filename)
      if fn not in files:
        files[fn] = {}
      files[fn]['train'] = train

      if fe == '.txt':
        files[fn]['caption'] = os.path.join(path, filename)
      elif fe == '.html':
        files[fn]['text'] = os.path.join(path, filename)
      elif fe == '.jpg':
        files[fn]['image'] = os.path.join(path, filename)
          
  insert_filenames(train_files, True, train_files_path)
  insert_filenames(test_files, False, test_files_path)
    
  try:

    # parse whole data into a new internal structure
    data = {}

    for fn, fa in files.items():
      data[fn] = {}
      #print(fn.encode('utf8', 'surrogateescape'))
      # just store path to image
      data[fn]['image'] = fa['image']

      data[fn]['train'] = fa['train']

      # read the caption
      with open(fa["caption"].encode('utf8', 'surrogateescape'), 'r', errors='ignore') as caption_file:
        data[fn]['caption'] = caption_file.read().strip()
        assert(len(data[fn]['caption']) > 0)

      # parse the text
      tree = etree.parse(fa["text"].encode('utf8', 'surrogateescape'), htmlParser)

      data[fn]['text'] = [] # list of tuples (title, paragraph)

      currTitle = []
      currPar = []
      for tag in tree.getroot().getchildren()[0].getchildren():
        if len(tag.getchildren()) > 0: # this is a paragraph which contains a multiline title
          if len(currTitle) > 0 or len(currPar) > 0:
            data[fn]['text'].append((currTitle, currPar))

            currTitle = []
            for ctag in tag.getchildren():
              assert ctag.tag == 'b'
              currTitle.append(ctag.text)
            currPar = []

        elif tag.text.strip() == '':
          continue
          
        elif tag.tag == 'b':

          if len(currTitle) > 0 or len(currPar) > 0:
            data[fn]['text'].append((currTitle, currPar))

          currTitle = [tag.text.strip()]
          currPar = []

        elif tag.tag == 'p':
          currPar += sent_detector.tokenize(tag.text.strip())
          
        else:
          raise Exception('Tag ' + tag.tag + ' not handled')

      if len(currTitle) > 0 or len(currPar) > 0:
        data[fn]['text'].append((currTitle, currPar))
        
      # store data[fn]
      json_line = json.dumps(data[fn])			
      OUTPUT_FILE.write(json_line + '\n')

  except Exception as e:
    traceback.print_exc()
    OUTPUT_FILE.close()
    sys.exit()
    
OUTPUT_FILE.close()

