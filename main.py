# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import io
import collections
import math
from PIL import Image
import requests


print(tf.__version__)

batch_size = 32
img_height = 256
img_width = 256

dataset_name = "aqua reimu face"
dataset_path = f"datasets/datasets/{dataset_name}"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_name,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_name,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)


data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.05),
    layers.experimental.preprocessing.RandomZoom(0.07),
  ]
)

"""
for i in range(1):
  plt.figure(figsize=(10, 10))
  for images, _ in train_ds.take(1):
    for i in range(9):
      augmented_images = data_augmentation(images)
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(augmented_images[0].numpy().astype("uint8"))
      plt.axis("off")
  plt.show()
#input()"""


num_classes = len(class_names)

model = tf.keras.Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1. / 255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

if os.path.isfile("model/checkpoint"):
  print("Loading weights")
  latest = tf.train.latest_checkpoint("model")
  model.load_weights(latest)
  print("Loaded weights")
else:
  print("Did not load weights!!")

epochs = 100

checkpoint_path = "model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

"""cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)"""

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
      save_weights_only=True,
      monitor='val_accuracy',
      mode='max',
      save_best_only=True)

classes = ["aqua", "fubuki", "makise kurisu", "megumin", "reimu hakurei", "rem"]

"""
for i in range(10):
  plt.figure(figsize=(10, 10))
  for images, _ in train_ds.take(1):
    for i in range(9):
      ax = plt.subplot(3, 420, i + 1)
      plt.imshow(images[0].numpy().astype("uint8"))
      plt.axis("off")
  plt.show()"""

if False:
  img_url = "https://img2.gelbooru.com/images/02/24/0224a423f22fa6855465db43c9ad76ac.jpg"
  img_bytes = requests.get(img_url).content
  #print(img_bytes)

  img = Image.open(io.BytesIO(img_bytes))
  img = img.convert('RGB')
  img = img.resize((img_width, img_height), Image.NEAREST)
  img = np.array(img)

  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)  # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )
  input()

if False:
  predictions = model.predict(val_ds)

  data = train_ds.take(len(predictions))
  for i in range(len(predictions)):
    for images, _ in train_ds.take(1):
      if _.numpy()[i] == np.argmax(predictions[i]):
        continue
      #print(_)
      #print(dir(_))
      #print(predictions[i])
      #print(max(predictions[i]))
      #print(np.argmax(predictions))
      #ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      #plt.axis("off")

      plt.xlabel(f"Guess: {np.argmax(predictions[i])} Real: {_.numpy()[i]}")
      plt.show()

  for i in range(len(predictions)):
    score = tf.nn.softmax(predictions[i])
    plt.figure(figsize=(10, 10))
    #print(val_ds)
    print(predictions[i])
    print(dir(data))
    print(data.asoutput())

    print(data)
    print(data[i])
    plt.imshow(data[i][0].numpy().astype("uint8"))
    plt.axis("off")
    plt.show()

AUTOTUNE = tf.data.experimental.AUTOTUNE


train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[cp_callback]
)

test_loss, test_acc = model.evaluate(val_ds, verbose=2)

print('\nTest accuracy:', test_acc)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

