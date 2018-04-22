#!/usr/bin/env python3
# In case annotation (using the wikiAnnotator) were collecting on different machines, there output files have to be merged. This is done script.

import argparse
import os
import json

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input1', type=str, help="A file containing wiki annotations (.jsonl file).")
  parser.add_argument('input2', type=str, help="Another file containing wiki annotations (.jsonl file).")
  parser.add_argument('output', type=str, help="Name of the jsonl file where the merged annotations should be stored.")
  parser.add_argument('--remove', type=bool, help="Whether or not the old files should be removed afterwards.", default=False)
  args = parser.parse_args()
  
  in_fn1 = args.input1
  in_fn2 = args.input2
  out_fn = args.output
  remove_old = args.remove

  if not os.path.isfile(in_fn1):
    raise(Exception('The file' + in_fn1 + 'does not exists'))
    
  if not os.path.isfile(in_fn2):
    raise(Exception('The file' + in_fn2 + 'does not exists'))
    
  if os.path.isfile(out_fn):
    print('The file %s already exists and will be overwritten.' % (out_fn))
  
  with open(in_fn1, 'r') as f:
    data1 = [json.loads(line) for line in f]
  with open(in_fn2, 'r') as f:
    data2 = [json.loads(line) for line in f]
  
  out_data = data1
  
  # Add all annotations from data2, that are not already part of the out_data.
  # Note, that (id, name, section) uniquely identify an annotation.
  for nd in data2:
    already_exists = False
    for d in data1: # assume that there are no doubles in data2
      if nd['id'] == d['id'] \
        and nd['name'] == d['name'] \
        and nd['section'] == d['section']:
        print('Annotation with (id=%d, name=%s, section=%s) exists in both sets.' % (d['id'], d['name'], d['section']))
        already_exists = True
        break
    if not already_exists:
      out_data.append(nd)
    
  with open(out_fn, 'w') as f:   
    for d in out_data:
      jsonLine = json.dumps(d)      
      f.write(jsonLine + '\n')
      
  if remove_old:
    if os.path.abspath(in_fn1) != os.path.abspath(out_fn):
      os.remove(in_fn1)
      print('Removed input file %s' % (in_fn1))
    if os.path.abspath(in_fn2) != os.path.abspath(out_fn):
      os.remove(in_fn2)
      print('Removed input file %s' % (in_fn2))
      
  
  print('Output file %s contains %d annotations' % (out_fn, len(out_data)))
  
