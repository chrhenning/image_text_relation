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

# Script to evaluate runnning training
#
# usage:
#  ./run_evaluation.sh

set -e

# Directory containing preprocessed data.
TFRECORD_DIR="${HOME}/autoencoder-input-win2"

# Directory to save the model.
MODEL_DIR="${HOME}/ImageTextRelation/word2vec/model"

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
export CUDA_VISIBLE_DEVICES=""

# Start training
"./evaluate.py" \
  --checkpoint_dir="${MODEL_DIR}/train" \
  --eval_dir="${MODEL_DIR}/eval" \
  --min_global_step=0 \
  --vocab_file="${TFRECORD_DIR}/word_counts.txt" \
