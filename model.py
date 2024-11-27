import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import io
import pandas as pd
import numpy as np


TRAINING_SIZE = 0.8
BATCH_SIZE = 16
NUM_EPOCHS = 15

data = pd.read_csv('data.csv')

label_map = {
    'rice': 0,
    'maize': 1,
    'chickpea': 2,
    'kidneybeans': 3,
    'pigeonpeas': 4,
    'mothbeans': 5,
    'mungbean': 6,
    'blackgram': 7,
    'lentil': 8,
    'pomegranate': 9,
    'banana': 10,
    'mango': 11,
    'grapes': 12,
    'watermelon': 13,
    'muskmelon': 14,
    'apple': 15,
    'orange': 16,
    'papaya': 17,
    'coconut': 18,
    'cotton': 19,
    'jute': 20,
    'coffee': 21,
}


features = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
labels = data['label']
labels = labels.map(label_map)

dataset = tf.data.Dataset.from_tensor_slices((features.values, labels.values))
dataset = dataset.shuffle(buffer_size=len(features), reshuffle_each_iteration=True)

total_size = len(features)
train_size = int(total_size * TRAINING_SIZE)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)
train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# print("Sample from training dataset:")
# for batch in train_dataset.take(1):
#     print(batch)

# print("\nSample from testing dataset:")
# for batch in test_dataset.take(1):
#     print(batch)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(features.shape[1],)),  
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(len(label_map), activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  
    metrics=['accuracy']
)
model.summary()

model.fit(
    train_dataset, 
    epochs=NUM_EPOCHS,
    validation_data = test_dataset
)

model.save('trained_model.keras')