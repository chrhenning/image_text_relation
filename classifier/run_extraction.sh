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

# Script to extract information from the model (such as article embeddings)
#
# Notice, the extracted embeddings use weights restored from the classifier model.
# If you would like to use the autoencoder weights instead, then shortly run the classifier 
# from an autoencoder checkpoint with flag train_autoencoder set to false 
# until a new classifier checkpoint was generated.
#
# usage:
#  ./run_extraction.sh

set -e

# Directory containing preprocessed data.
#TFRECORD_DIR="${HOME}/classifier-input"
TFRECORD_DIR="${HOME}/classifier-input-simple"

CHECKPOINT_DIR="${HOME}/ImageTextRelation/classifier/model_mi-m_sc-r/best/model.ckpt-311"

# Directory to save the model.
MODEL_DIR="${HOME}/ImageTextRelation/classifier/extracted"

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
export CUDA_VISIBLE_DEVICES=""

# Start extraction of training samples
"./extraction.py" \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-00016" \
  --checkpoint_dir="${CHECKPOINT_DIR}" \
  --extraction_dir="${MODEL_DIR}" \
  --extraction_file="train_samples" \
  
# Start extraction of test samples
"./extraction.py" \
  --input_file_pattern="${TFRECORD_DIR}/test-?????-of-00002" \
  --checkpoint_dir="${CHECKPOINT_DIR}" \
  --extraction_dir="${MODEL_DIR}" \
  --extraction_file="test_samples" \
