import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(5,5),activation='relu',padding='same',kernel_initializer='he_normal',input_shape=(120,120,3)),
    tf.keras.layers.MaxPool2D((3,3),padding='same'),
    tf.keras.layers.Conv2D(128,(5,5),activation="relu",padding='same',kernel_initializer='he_normal'),
    tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding='same',kernel_initializer='he_normal'),
    tf.keras.layers.MaxPool2D((3,3)),
    tf.keras.layers.Conv2D(256,(5,5),activation="relu",kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256,(5,5),activation="relu",kernel_initializer='he_normal'),
    tf.keras.layers.MaxPool2D((3,3),strides=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=250,activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=250,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units = 26,activation='softmax')
])

print(model.summary())

# reducing learning rate to prevent the saturation of loss
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    mode='min',
    min_lr=0.001
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss= tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]
    )

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    validation_split=0.3,
)  


train_gen = datagen.flow_from_directory(
    "C:/Users/LENEVO/Desktop/pps dataset",
    target_size=(120,120),
    batch_size=50,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    subset='training',
)

val_gen = datagen.flow_from_directory(
    "C:/Users/LENEVO/Desktop/pps dataset",
    target_size=(120,120),
    batch_size=50,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    subset='validation',
)

history = model.fit(
    train_gen,
    epochs=12,
    callbacks=[reduce_lr],
    steps_per_epoch=132,    # no.of data samples of training set/no. of batches
    validation_data=val_gen,
    validation_steps=56,    # no.of data samples of validation set/no. of batches
    )

model.save("mark7.h5")

# plotting the accuracy and loss
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

plot_labels = ['Accuracy', 'Loss']
plot_styles = ['-', '--']

for i, metric in enumerate(['accuracy', 'loss']):
    train_metric = history.history[metric]
    test_metric = history.history['val_' + metric]
    axs[i].plot(train_metric, label='Training ' + metric.capitalize(), linestyle=plot_styles[0])
    axs[i].plot(test_metric, label='Testing ' + metric.capitalize(), linestyle=plot_styles[1])
    axs[i].set_xlabel('Epochs')
    axs[i].set_ylabel(plot_labels[i])
    axs[i].legend()

plt.tight_layout()
plt.show()
