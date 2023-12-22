import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

model = Sequential([
    keras.layers.Conv2D(724,(3,3),activation='relu',padding = 'same',input_shape=(32,32,3)),
    keras.layers.MaxPool2D((2,2),padding = 'same'),
    keras.layers.Conv2D(500,(3,3),activation="relu",padding = 'same'),
    keras.layers.MaxPool2D((2,2),padding = 'same'),
    keras.layers.Conv2D(250,(3,3),activation="relu",padding = 'same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2,2),padding = 'same'),
    keras.layers.Flatten(),
    keras.layers.Dense(units=512,activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(units = 26,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss= tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
    )

print(model.summary())

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
    # "C:/Users/LENEVO/Desktop/dataset/",
    "C:/Users/LENEVO/Desktop/pps dataset",
    target_size=(32,32),
    batch_size=50,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    subset='training',
    seed=100,
)

val_gen = datagen.flow_from_directory(
    # "C:/Users/LENEVO/Desktop/dataset/",
    "C:/Users/LENEVO/Desktop/pps dataset",    
    target_size=(32,32),
    batch_size=50,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    subset='validation',
    seed=100,
)

history = model.fit(train_gen,
    epochs=10,
    steps_per_epoch=150,  # no.of data samples/no. of batches
    validation_data=val_gen,
    validation_steps=37,
    )

model.save("mark1.h5")

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

# # Get the model's predictions on the test set
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true_classes = np.argmax(y_test, axis=1)
# print()