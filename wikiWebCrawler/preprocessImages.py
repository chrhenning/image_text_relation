#!/usr/bin/env python3
# This tool can be used to preprocess already downloaded images.
# SVG graphics are rasterized
# The first frame of a GIF is used
# All graphics are converted to jpg. Transparency is handled by applying alpha blending (composition) with a white background

import argparse
import os
import cv2
import json
import numpy as np
import cairosvg
from PIL import Image
import tempfile
import logging
import sys
import traceback

logging.basicConfig(filename=os.path.join(tempfile.gettempdir(), 'image_preprocessing.log'),level=logging.DEBUG)

# use alpha blending to lay the image onto a white background (computation is in-place)
# description: http://stackoverflow.com/questions/2049230/convert-rgba-color-to-rgb
def convertRGBAintoRGB(img):
  for i in range(3):
    img[:,:,i] = (255-img[:,:,3])*1 + (img[:,:,3]* (1.0/255.0)) * img[:,:,i]
  img = np.delete(img, 3,2) # delete alpha channel
    
  return img
  
# resize the image, such that the original ratio is kept (the smaller dimension will have the user defined image size)
def resizeImage(img):
  global IMG_SIZE
  
  h = img.shape[0]
  w = img.shape[1]

  nh = -1
  nw = -1

  if h > w:
    nw = IMG_SIZE
    nh = int(float(h/w) * IMG_SIZE)
  
  else:
    nh = IMG_SIZE
    nw = int(float(w/h) * IMG_SIZE)

  img = cv2.resize(img, (nw, nh), interpolation = cv2.INTER_CUBIC)
  
  return img

# images is an array of image objects
# 1. convert all images into a 3 channel image (RGB)
# 2. shrink size of image
def preprocess_images(images):
  global DATASET_PATH, PROCESSED_IMAGES, REJECTED_IMAGES
  tmp_png = os.path.join(tempfile.gettempdir(), 'tmp_img.png')

  for img in images:
    if 'imgpath' not in img: # image wasn't downloaded, probably because it wasn't supported by enough surrounding text
      continue

    ip = os.path.join(DATASET_PATH, img['imgpath'])
    file_name, file_extension = os.path.splitext(ip)
  
    img_matrix = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
    
    # it is probably a gif or svg graphic (we convert it into a png graphic
    if img_matrix is None:
      if file_extension.lower() == '.gif':
          # get first frame of gif image
          im = Image.open(ip)
          if 'transparency' in im.info:
            transparency = im.info['transparency'] 
            im.save(tmp_png, transparency=transparency)
          else:
            im.save(tmp_png)
          im

          img_matrix = cv2.imread(tmp_png, cv2.IMREAD_UNCHANGED)

      elif file_extension.lower() == '.svg':
        try:
          cairosvg.svg2png(url=ip, write_to=tmp_png)
        except:
          logging.warning('SVG \'' + ip + '\' could not be converted to png.') 
          REJECTED_IMAGES += 1
          continue
        
        img_matrix = cv2.imread(tmp_png, cv2.IMREAD_UNCHANGED)    

    if img_matrix is not None:
      PROCESSED_IMAGES += 1
      
      # if alpha channle exists
      if len(img_matrix.shape) == 3 and img_matrix.shape[2] == 4:
        img_matrix = convertRGBAintoRGB(img_matrix)

      img_matrix = resizeImage(img_matrix)

      # where the new image should be saved at
      nip = file_name + '.jpg'
      
      # change image object
      # state, what the original imageformat is
      if file_extension.lower() in ['.jpg', '.jpeg']:
        img['origformat'] = 'jpg'
      elif file_extension.lower() == '.png':
        img['origformat'] = 'png'
      elif file_extension.lower() == '.gif':
        img['origformat'] = 'gif'
      elif file_extension.lower() == '.svg':
        img['origformat'] = 'svg'
      else:
        logging.warning('Format \'' + file_extension + '\' not handled') 
        REJECTED_IMAGES += 1
        continue
      
      new_img_path, _ = os.path.splitext(img['imgpath'])
      img['imgpath'] = new_img_path + '.jpg'
      
      # remove old image file (since it gets not overwritten)
      if ip != nip:
        os.remove(ip) 
      
      # save image as jpg
      cv2.imwrite(nip, img_matrix) 

    else:
      REJECTED_IMAGES += 1
      
      logging.warning('Image \'' + ip + '\' was rejected') 



# recursively iterate through sectins and subsection and preprocess all occuring images
def iterate_sections(sections):
  for sec in sections:
    preprocess_images(sec['images'])
    iterate_sections(sec['subsections'])

global DATASET_PATH
global PROCESSED_IMAGES
global REJECTED_IMAGES
global IMG_SIZE

DATASET_PATH = './simplewiki-dataset'
PROCESSED_IMAGES = 0
REJECTED_IMAGES = 0
IMG_SIZE = 346

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, help="Directory, that contains the dataset", default=DATASET_PATH)
  parser.add_argument('--size', type=int, help="The script will keep the aspect ratio, but the smaller dimension will be fit to this size.", default=IMG_SIZE)
  args = parser.parse_args()

  DATASET_PATH = args.data
  IMG_SIZE = args.size

  dataset_file = os.path.join(DATASET_PATH, 'articles.jsonl')
  # we write the modified json objects into this file (new filenames)
  dataset_file_new = os.path.join(DATASET_PATH, 'articles_tmp.jsonl')

  if not os.path.isdir(DATASET_PATH):
    raise(Exception('Directory ' + DATASET_PATH + ' does not exist.'))

  if not os.path.isfile(dataset_file):
    raise(Exception('File ' + dataset_file + ' does not exist.'))
    
  try:
    fw = open(dataset_file_new, 'w')

    iter = 0
    # for each wiki article
    for line in open(dataset_file, 'r'):
      article = json.loads(line)

      preprocess_images(article['images'])

      iterate_sections(article['sections'])

      #if iter == 10:
      #  break

      iter += 1
      if not iter % 50:
        logging.info('Processed %d  articles with %d images (%d images have been rejected)' % (iter, PROCESSED_IMAGES, REJECTED_IMAGES)) 
        
      # write modified data into temporary file
      json_line = json.dumps(article)			
      fw.write(json_line + '\n')
		
  except Exception as e:
    traceback.print_exc()
    logging.exception('Dataset is not overwritten. However, image preprocessing steps cannot be reverted.')
    fw.close()
    sys.exit()
    
  logging.info('Summary: processed %d articles with %d images (%d images have been rejected)' % (iter, PROCESSED_IMAGES, REJECTED_IMAGES)) 

  fw.close()

  # overwrite old dataset with new one
  os.rename(dataset_file_new, dataset_file)
