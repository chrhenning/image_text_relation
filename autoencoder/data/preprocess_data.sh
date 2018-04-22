#!/bin/bash
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

# Script to start the preprocessing of all three datasets: mscoco, simple-wiki, bbc news
#
# The outputs of this script are sharded TFRecord files containing serialized
# SequenceExample protocol buffers. See buid_data.py for details of how
# the SequenceExample protocol buffers are constructed.
#
# usage:
#  ./preprocess_data.sh

# directories of datasets
COCO_DIR="/mnt/data/mscoco"
WIKI_DIR="${HOME}/ImageTextRelation/wikiDataset/simplewiki-dataset"
BBC_DIR="${HOME}/ImageTextRelation/bbcAnnotator/src/data"

set -e

if [ -z "$1" ]; then
  echo "usage preprocess_data.sh [data dir]"
  exit
fi

# Create the output directories.
OUTPUT_DIR="${1%/}" # strip tailing slash
mkdir -p "${OUTPUT_DIR}"

TRAIN_IMAGE_DIR_COCO="${COCO_DIR}/train2014"
VAL_IMAGE_DIR_COCO="${COCO_DIR}/val2014"
TRAIN_CAPTIONS_FILE_COCO="${COCO_DIR}/annotations/captions_train2014.json"
VAL_CAPTIONS_FILE_COCO="${COCO_DIR}/annotations/captions_val2014.json"

# Build TFRecords of the image data.
"./build_data.py" \
  --train_image_dir_coco="${TRAIN_IMAGE_DIR_COCO}" \
  --val_image_dir_coco="${VAL_IMAGE_DIR_COCO}" \
  --train_captions_file_coco="${TRAIN_CAPTIONS_FILE_COCO}" \
  --val_captions_file_coco="${VAL_CAPTIONS_FILE_COCO}" \
  --wiki_data_dir="${WIKI_DIR}/wiki-samples.jsonl" \
  --bbc_data_dir="${BBC_DIR}/bbc-samples.jsonl" \
  --output_dir="${OUTPUT_DIR}" \
  --word_counts_output_file="${OUTPUT_DIR}/word_counts.txt" \
  --lemmatize_tokens=1 \
  --skip_window=5 \
  --num_skips=8 \
