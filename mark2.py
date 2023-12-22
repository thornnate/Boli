import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(5,5),activation='relu',input_shape=(120,120,3)),
    tf.keras.layers.MaxPool2D((3,3),strides=2),
    tf.keras.layers.Conv2D(64,(5,5),activation="relu"),
    tf.keras.layers.MaxPool2D((3,3),strides=2),
    tf.keras.layers.Conv2D(128,(5,5),activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((3,3),strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=243,activation='relu'),
    tf.keras.layers.Dense(units=243,activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units = 26,activation='softmax')
])

print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss= tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
    )

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.1, # Randomly zoom image 
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    validation_split=0.2,
)  

train_gen = datagen.flow_from_directory(
    "C:/Users/LENEVO/Desktop/dataset/",
    target_size=(120,120),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    subset='training',
    seed=100,
)

val_gen = datagen.flow_from_directory(
    "C:/Users/LENEVO/Desktop/dataset/",
    target_size=(120,120),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    subset='validation',
    seed=100,
)

history = model.fit(train_gen,
    epochs=10,
    steps_per_epoch=89,  # no.of data samples/no. of batches
    validation_data=val_gen,
    validation_steps=22,
    )

model.save("mark2.h5")

# plotting the accuracy and loss
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Define plot labels and styles
plot_labels = ['Accuracy', 'Loss']
plot_styles = ['-', '--']

# Plot training and testing accuracy/loss
for i, metric in enumerate(['accuracy', 'loss']):
    train_metric = history.history[metric]
    test_metric = history.history['val_' + metric]
    axs[i].plot(train_metric, label='Training ' + metric.capitalize(), linestyle=plot_styles[0])
    axs[i].plot(test_metric, label='Testing ' + metric.capitalize(), linestyle=plot_styles[1])
    axs[i].set_xlabel('Epochs')
    axs[i].set_ylabel(plot_labels[i])
    axs[i].legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# add a test part later

