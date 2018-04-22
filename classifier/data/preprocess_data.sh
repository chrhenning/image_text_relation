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
#COCO_DIR="/mnt/data/mscoco"
COCO_FILE="${HOME}/ImageTextRelation/annotations/coco-anno-samples.jsonl"
WIKI_DIR="${HOME}/ImageTextRelation/wikiDataset/simplewiki-dataset"
BBC_DIR="${HOME}/ImageTextRelation/bbcAnnotator/src/data"
AUTOENCODER_INPUT="${HOME}/autoencoder-input"

set -e

if [ -z "$1" ]; then
  echo "usage preprocess_data.sh [data dir]"
  exit
fi

# Create the output directories.
OUTPUT_DIR="${1%/}" # strip tailing slash
mkdir -p "${OUTPUT_DIR}"

# The classifier only consideres samples from the validation data
#TRAIN_IMAGE_DIR_COCO="${COCO_DIR}/train2014"
#VAL_IMAGE_DIR_COCO="${COCO_DIR}/val2014"

# Build TFRecords of the image data.
"./build_data.py" \
  --coco_sample_file="${COCO_FILE}" \
  --wiki_sample_file="${WIKI_DIR}/wiki-anno-samples.jsonl" \
  --bbc_sample_file="${BBC_DIR}/bbc-anno-samples.jsonl" \
  --output_dir="${OUTPUT_DIR}" \
  --word_counts_input_file="${AUTOENCODER_INPUT}/word_counts.txt" \
  --lemmatize_tokens=1 \
  --only_photographs=false \
