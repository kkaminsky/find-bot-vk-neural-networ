from __future__ import absolute_import, division, print_function, unicode_literals
from flask import jsonify, request
from flask import Flask
import numpy as np
import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

app = Flask(__name__)

URL = 'file:///C:/Users/Konstantin/find_bot_target_201906021630.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    dataframe[['societies_flag']] = (dataframe[['societies_flag']]).astype(int)
    dataframe.pop('id')
    labels = dataframe.pop('bot_flag').astype(int)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 32  # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of uniq_cities:', feature_batch['uniq_cities'])
    print('A batch of targets:', label_batch)


feature_columns = []

for header in ['groups_count', 'uniq_cities', 'triggers_words_count', 'photos_count', 'max_percent_photos_in_day',
               'posts_count', 'max_percent_posts_in_day', 'posts_without_views', 'group_without_photo_count']:
    feature_columns.append(feature_column.numeric_column(header))

for header in ['societies_flag']:
    feature_columns.append(
        feature_column.indicator_column(feature_column.categorical_column_with_identity(key=header, num_buckets=2)))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 10
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=50)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)


@app.route('/')
def index():
    return 'Server Works!'


@app.route('/greet', methods=['POST'])
def say_hello():
    print(request.get_json())
    predict = model.predict(request.get_json()['arr'])
    return "{0:.2f}".format(float(predict[0][0]))