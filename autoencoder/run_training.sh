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

# Script to start the training of the autoencoder
#
# usage:
#  ./run_training.sh

set -e

# Directory containing preprocessed data.
TFRECORD_DIR="${HOME}/autoencoder-input"

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="${HOME}/ImageTextRelation/autoencoder/data/inception_v3.ckpt"

# Word2Vec checkpoint path.
WORD2VEC_CHECKPOINT="${HOME}/ImageTextRelation/word2vec/model/train"

# Directory to save the model.
MODEL_DIR="${HOME}/ImageTextRelation/autoencoder/model"

# Start training
"./train.py" \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-00512" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --embedding_checkpoint_dir="${WORD2VEC_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=true \
  --train_embeddings=true \
  --number_of_steps=1000000
