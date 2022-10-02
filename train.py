#!/bin/python

seed = 42

import os
os.environ['PYTHONHASHSEED'] = str(seed)

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed

from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import metrics
from efficientnet_lite import EfficientNetLiteB0

def get_images_from_path(dataset_path):
    """ Get images from path and normalize them applying channel-level normalization. """

    # loading all images in one large batch
    tf_eval_data = tf.keras.utils.image_dataset_from_directory(dataset_path, image_size=input_shape[:2], shuffle=False,
                                                               batch_size=100000)

    # extract images and targets
    for tf_eval_images, tf_eval_targets in tf_eval_data:
        break

    return tf.convert_to_tensor(tf_eval_images), tf_eval_targets

def evaluate_model(features, labels):
    
    predictions = np.zeros(len(labels), dtype=np.int8)
    
    for e, (image, target) in enumerate(zip(features, labels)):
        image = np.expand_dims(np.array(image), axis=0)
        output = model.predict(image)
        predictions[e] = np.squeeze(output).argmax()
    
    score_keras = 1 - metrics.cohen_kappa_score(labels.numpy(), predictions)
    print("Score:",score_keras)
    
    return predictions

input_shape = (200, 200, 3)   # input_shape is (height, width, number of channels) for images
num_classes = 8
model = EfficientNetLiteB0(classes=num_classes, weights=None, input_shape=input_shape, classifier_activation=None)

dataset_path="../ops_sat_competiton_official"

#Loading dataset
x_train, y_train = get_images_from_path(dataset_path)

x_train = x_train.numpy().astype(np.int32)
y_train = y_train.numpy()

x_train_rot90_ccw = np.rot90(x_train, axes=(1,2))
x_train_augmented = np.vstack([x_train, x_train_rot90_ccw])
y_train_augmented = np.concatenate([y_train, y_train])

x_train_rot180 = np.rot90(np.rot90(x_train, axes=(1,2)), axes=(1,2))
x_train_augmented = np.vstack([x_train_augmented, x_train_rot180])
y_train_augmented = np.concatenate([y_train_augmented, y_train])

x_train_rot90_cw = np.rot90(x_train, axes=(2,1))
x_train_augmented = np.vstack([x_train_augmented, x_train_rot90_cw])
y_train_augmented = np.concatenate([y_train_augmented, y_train])

x_train_flipud = np.array([np.flipud(i) for i in x_train])
x_train_augmented = np.vstack([x_train_augmented, x_train_flipud])
y_train_augmented = np.concatenate([y_train_augmented, y_train])

x_train_fliplr = np.array([np.fliplr(i) for i in x_train])
x_train_augmented = np.vstack([x_train_augmented, x_train_fliplr])
y_train_augmented = np.concatenate([y_train_augmented, y_train])

x_train_rot90_ccw_flipud = np.array([np.flipud(i) for i in x_train_rot90_ccw])
x_train_augmented = np.vstack([x_train_augmented, x_train_rot90_ccw_flipud])
y_train_augmented = np.concatenate([y_train_augmented, y_train])

x_train_rot90_cw_flipud = np.array([np.flipud(i) for i in x_train_rot90_cw])
x_train_augmented = np.vstack([x_train_augmented, x_train_rot90_cw_flipud])
y_train_augmented = np.concatenate([y_train_augmented, y_train])

x_train_augmented, y_train_augmented = shuffle(x_train_augmented, y_train_augmented, random_state=seed)

factor = 0.1
share = 0.5
max_index = int(round(x_train_augmented.shape[0] * share, 0))

x_train_noise = np.array([np.round(i + (np.random.normal(i, factor * i)), 0) for i in x_train_augmented[:max_index]]).astype(np.int32)
#x_train_augmented = np.vstack([x_train_augmented, x_train_noise])
#y_train_augmented = np.concatenate([y_train_augmented, y_train_augmented[:max_index]])

x_train_augmented, y_train_augmented = shuffle(x_train_augmented, y_train_augmented, random_state=seed)

x_train = tf.convert_to_tensor(x_train_augmented)
y_train = tf.convert_to_tensor(y_train_augmented, dtype=np.int32)

print('x train shape: {}; y train shape: {}'.format(x_train.shape, y_train.shape))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseCategoricalAccuracy()])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history=model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=8, callbacks=[callback], validation_split=0.2, shuffle=False)

predictions = evaluate_model(x_train, y_train)

print('Accuracy: {:0.3f}'.format(round(metrics.accuracy_score(y_train, predictions), 3)))

# What proportion of positive identifications was actually correct?
print('Precision: {:0.3f}'.format(round(metrics.precision_score(y_train, predictions, average='micro'), 3)))

# What proportion of actual positives was identified correctly?
print('Recall: {:0.3f}'.format(round(metrics.recall_score(y_train, predictions, average='micro'), 3)))

print('F1-Score: {:0.3f}'.format(round(metrics.f1_score(y_train, predictions, average='micro'), 3)))

print(metrics.classification_report(y_train, predictions))

#model.save_weights('v25.h5')