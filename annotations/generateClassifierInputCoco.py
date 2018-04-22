#!/usr/bin/env python3
# This script generates the input for the SC and MI classifier (which is then used to generate the TFRecord File) for the mscoco dataset.
# Therefore, it takes a userdefined number of images and a selection of random captions from this image to generate a sample, that is considered as
# being highly semantically correlated (sc = 1) with the text only stating visual obvious information from the image (mi = 6)

import argparse
import os
import json
import random
random.seed(42)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--images', type=str, help="Directory containing the mscoco images", default='/mnt/data/mscoco/val2014')
  parser.add_argument('--captions', type=str, help="File containing the image captions", default='/mnt/data/mscoco/annotations/captions_val2014.json')
  parser.add_argument('--output', type=str, help="Name of the jsonl file where the merged annotations should be stored.", default='coco-anno-samples.jsonl')
  parser.add_argument('--number', type=int, help="How many image-caption pairs should be stored in the output file (randomly selected)", default=100)
  parser.add_argument('--label', type=int, help="The MI label that shall be assigned to the generated samples.", default=6)
  args = parser.parse_args()
  
  images_dir = args.images
  captions_fn = args.captions
  out_fn = args.output
  num_samples = args.number
  mi_label = args.label

  if not os.path.isdir(images_dir):
    raise(Exception('The directory ' + images_dir + ' does not exists'))
    
  if not os.path.isfile(captions_fn):
    raise(Exception('The file ' + captions_fn + ' does not exists'))
    
  if os.path.isfile(out_fn):
    print('The file %s already exists and will be overwritten.' % (out_fn))
    
  if mi_label not in range(8):
    raise(Exception('Label ' + str(mi_label) + ' is not a valid MI label'))
  
  with open(captions_fn, "r") as f:
    captions = json.load(f)

  id_to_filename = [(x["id"], x["file_name"]) for x in captions["images"]]
  id_to_captions = {}
  for annotation in captions["annotations"]:
    image_id = annotation["image_id"]
    caption = annotation["caption"]
    id_to_captions.setdefault(image_id, [])
    id_to_captions[image_id].append(caption)
    
  # generate dataset, that contains tuples (abs img path, captions)
  dataset = []
  for image_id, base_filename in id_to_filename:
    filename = os.path.join(images_dir, base_filename)
    captions = [c for c in id_to_captions[image_id]]
    dataset.append((filename, captions))
  
  random.shuffle(dataset)
  dataset = dataset[:num_samples]

  # use tuples to build new samples
  samples_anno = []
  
  for tup in dataset:
    random.shuffle(tup[1])
    # select a random number of captions, that form a variable sized text (max-length is 5)
    captions = tup[1][:random.randint(1,len(tup[1])-1)]
    assert(len(captions) >= 1)
  
    new_sample = {}
    new_sample['imgpath'] = tup[0]
    new_sample['text'] = captions
    new_sample['annotation'] = {}
    new_sample['annotation']['mi'] = mi_label
    new_sample['annotation']['sc'] = 1
    new_sample['annotation']['type'] = 'Photograph'
    new_sample['annotation']['snippets'] = []
    samples_anno.append(new_sample)
    
  with open(out_fn, 'w') as f:   
    for d in samples_anno:
      jsonLine = json.dumps(d)      
      f.write(jsonLine + '\n')

  print('Output file %s contains %d samples' % (out_fn, len(samples_anno)))

