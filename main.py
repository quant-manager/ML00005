#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 James James Johnson. All Rights Reserved.
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

"""
@author: James J Johnson
@url: https://www.linkedin.com/in/james-james-johnson/
"""

# Install this package if running on your local machine

# pip install protobuf==3.20.3

# pip install -q tensorflow-datasets

# pip install protobuf==4.23.3

# https://stackoverflow.com/questions/72485953/protobuf-incompatibility-when-installing-tensorflow-model-garden
# Google introduced a new breaking change in protobuf-4.21.0. Anything later than that will not work.
# You need to install protobuf 3.20.x or earlier (like version 3.20.3.)


###############################################################################
# Load packages

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

###############################################################################
# Load data

dataset, info = tfds.load(
    'imdb_reviews/subwords8k',
    with_info=True,
    as_supervised=True)
tokenizer = info.features['text'].encoder

###############################################################################
# Define hyperparameters

buffer_size = 10000
batch_size = 256
embedding_dim = 64
lstm_dim = 64
dense_dim = 64
num_epochs = 10

###############################################################################
# Shuffle and pad data in datasets

train_data, validation_data = dataset['train'], dataset['test'], 
train_dataset = train_data.shuffle(buffer_size)
train_dataset = train_dataset.padded_batch(batch_size)
validation_dataset = validation_data.padded_batch(batch_size)

###############################################################################
# Define and compile model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim)),
    tf.keras.layers.Dense(dense_dim, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
print(model.summary())
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

###############################################################################
# Fit model

history = model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=validation_dataset,
    )

###############################################################################
# Plot model fitting history

def plot_history(history, metric_name):
    plt.plot(history.history[metric_name])
    plt.plot(history.history['val_' + metric_name])
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend([metric_name, 'val_' + metric_name])
    plt.show()

plot_history(history, "loss")
plot_history(history, "accuracy")
plt.savefig("fitting_history.pdf", format="pdf", bbox_inches="tight")
