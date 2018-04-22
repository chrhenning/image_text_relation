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

# Script to predict labels for given samples.
#
# Notice, that the script is not well implemented and hard to use (works only for validation coco
# samples at the moment). It has been implemented to extract article_embeddings and predictions for
# samples used to investigate image-sentence retrieval results
#
# usage:
#  ./run_prediction.sh

set -e

COCO_VAL_DIR="/mnt/data/mscoco/val2014"

#CHECKPOINT_DIR="${HOME}/ImageTextRelation/classifier/model/train"
CHECKPOINT_DIR="${HOME}/ImageTextRelation/classifier/model_mi-m_sc-r/best/model.ckpt-311"

# Directory to save the results.
RESULT_DIR="${HOME}/ImageTextRelation/classifier/predictions"

# Directory that contains a pickle dump with the samples that should be considered.
INPUT_DIR="${HOME}/ImageTextRelation/order_embeddings/data/top_candidates.json"

# Directory containing vocab file.
TFRECORD_DIR="${HOME}/autoencoder-input"

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
#export CUDA_VISIBLE_DEVICES=""

# Start prediction
"./predict_isr_candidates.py" \
  --checkpoint_dir="${CHECKPOINT_DIR}" \
  --prediction_dir="${RESULT_DIR}" \
  --data_path="${INPUT_DIR}" \
  --coco_val_path="${COCO_VAL_DIR}" \
  --vocab_file="${TFRECORD_DIR}/word_counts.txt" \
  
