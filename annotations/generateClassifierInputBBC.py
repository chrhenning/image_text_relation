#!/usr/bin/env python3
# This script generates the input for the SC and MI classifier (which is then used to generate the TFRecord File.
# Therefore, if takes the text-image tuples, which are the input for the autoencoder, and the annotations from the
# BBC annotator and merges them into new samples. These are stored as new jsonl file.

import argparse
import os
import json

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--samples', type=str, help="A file containing the image-text pairs of the BBC dataset (.jsonl file).", default='bbc-samples.jsonl')
  parser.add_argument('--annotations', type=str, help="Another file containing BBC annotations (.jsonl file).", default='bbcAnnotations.jsonl')
  parser.add_argument('--output', type=str, help="Name of the jsonl file where the merged annotations should be stored.", default='bbc-anno-samples.jsonl')
  args = parser.parse_args()
  
  samples_fn = args.samples
  annotations_fn = args.annotations
  out_fn = args.output

  if not os.path.isfile(samples_fn):
    raise(Exception('The file ' + samples_fn + ' does not exists'))
    
  if not os.path.isfile(annotations_fn):
    raise(Exception('The file ' + annotations_fn + ' does not exists'))
    
  if os.path.isfile(out_fn):
    print('The file %s already exists and will be overwritten.' % (out_fn))
  
  with open(samples_fn, 'r') as f:
    samples = [json.loads(line) for line in f]
  with open(annotations_fn, 'r') as f:
    annotations = [json.loads(line) for line in f]
    print('Annotations file %s contains %d annotations' % (annotations_fn, len(annotations)))
  
  samples_tuples = []
  
  # match an image-text pair with its corresponding annotation
  for anno in annotations:
    for pair in samples:
        if anno['train'] == pair['train']:
            file_name, _ = os.path.splitext(os.path.basename(pair['image']))
            if file_name == anno['name']:
                samples_tuples.append((pair, anno))

  # use tuples to build new samples
  samples_anno = []
  
  for tup in samples_tuples:
    if tup[1]['mi'] >= 0 and tup[1]['mi'] <= 7 \
      and tup[1]['sc'] >= -1 and tup[1]['sc'] <= 1:
      new_sample = tup[0]  
      new_sample['annotation'] = {}
      new_sample['annotation']['mi'] = tup[1]['mi']
      new_sample['annotation']['sc'] = tup[1]['sc']
      new_sample['annotation']['type'] = tup[1]['type']
      new_sample['annotation']['snippets'] = tup[1]['snippets']
      samples_anno.append(new_sample)
    else:
      print('Skipping invalid annotation')
      print(tup[1]) 
    
  with open(out_fn, 'w') as f:   
    for d in samples_anno:
      jsonLine = json.dumps(d)      
      f.write(jsonLine + '\n')

  print('Output file %s contains %d samples' % (out_fn, len(samples_anno)))
  
