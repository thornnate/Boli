# resnet model

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# identity block

def identity_block(X, f, filters, initializer=random_uniform):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X) # Default axis
    X = Activation('relu')(X)
    
    ### START CODE HERE
    ## Second component of main path (≈3 lines)
    ## Set the padding = 'same'
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    ## Third component of main path (≈2 lines)
    ## Set the padding = 'valid'
    X = Conv2D(filters = F3, kernel_size=1, strides=(1,1), padding = 'valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    
    ## Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    ### END CODE HERE

    return X

# convolution block

def convolutional_block(X, f, filters, s = 2, initializer=glorot_uniform):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                   also called Xavier uniform initializer.
    
    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    
    # First component of main path glorot_uniform(seed=0)
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    ### START CODE HERE
    
    ## Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size=f, strides=(1,1), padding='same',kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    ## Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=1, strides=(1,1), padding='valid',kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    
    ##### SHORTCUT PATH ##### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=(s,s), padding='valid', kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    
    ### END CODE HERE

    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X
    
# 50 layers ResNet model

def ResNet50(input_shape = (120, 120, 3), classes = 26, training=False):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    ### START CODE HERE
    
    # Use the instructions above in order to implement all of the Stages below
    # Make sure you don't miss adding any required parameter
    
    ## Stage 3 (≈4 lines)
    # `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X, f = 3,filters = [128,128,512], s = 2)
    
    # the 3 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, f = 3, filters = [128,128,512])
    X = identity_block(X, f = 3, filters = [128,128,512])
    X = identity_block(X, f = 3, filters = [128,128,512])

    # Stage 4 (≈6 lines)
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X, f = 3 , s = 2 , filters = [256,256,1024])
    
    # the 5 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X , f = 3 , filters = [256,256,1024])
    X = identity_block(X , f = 3 , filters = [256,256,1024])
    X = identity_block(X , f = 3 , filters = [256,256,1024])
    X = identity_block(X , f = 3 , filters = [256,256,1024])
    X = identity_block(X , f = 3 , filters = [256,256,1024])

    # Stage 5 (≈3 lines)
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X ,f = 3 , s = 2 , filters = [512,512,2048])
    
    # the 2 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, f = 3 , filters = [512,512,2048])
    X = identity_block(X, f = 3 , filters = [512,512,2048])

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D()(X)"
    X = AveragePooling2D()(X)
    
    ### END CODE HERE

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X)

    return model

tf.keras.backend.set_learning_phase(True)

model = ResNet50(input_shape = (120, 120, 3), classes = 26)
print(model.summary())

# np.random.seed(1)
# tf.random.set_seed(2)
# opt = tf.keras.optimizers.Adam(learning_rate=0.00015)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# # Normalize image vectors
# X_train = X_train_orig / 255.
# X_test = X_test_orig / 255.


# compiling the model
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(), # learning_rate=0.001
#     loss= tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#     metrics=['accuracy']
#     )

# datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.1,
#     rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
#     zoom_range = 0.1, # Randomly zoom image 
#     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#     validation_split=0.2,
# )  

# train_gen = datagen.flow_from_directory(
#     "C:/Users/LENEVO/Desktop/dataset/",
#     target_size=(64,64),
#     batch_size=32,
#     color_mode='rgb',
#     class_mode='categorical',
#     shuffle=True,
#     subset='training',
#     seed=100,
# )

# val_gen = datagen.flow_from_directory(
#     "C:/Users/LENEVO/Desktop/dataset/",
#     target_size=(64,64),
#     batch_size=32,
#     color_mode='rgb',
#     class_mode='categorical',
#     shuffle=True,
#     subset='validation',
#     seed=100,
# )

# # training model
# history = model.fit(train_gen,
#     epochs=10,
#     steps_per_epoch=72,  # no.of data samples/no. of batches
#     validation_data=val_gen,
#     validation_steps=18,
#     )

# model.save("mark3.h5")

# # plotting the accuracy and loss
# fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# # Define plot labels and styles
# plot_labels = ['Accuracy', 'Loss']
# plot_styles = ['-', '--']

# # Plot training and testing accuracy/loss
# for i, metric in enumerate(['accuracy', 'loss']):
#     train_metric = history.history[metric]
#     test_metric = history.history['val_' + metric]
#     axs[i].plot(train_metric, label='Training ' + metric.capitalize(), linestyle=plot_styles[0])
#     axs[i].plot(test_metric, label='Testing ' + metric.capitalize(), linestyle=plot_styles[1])
#     axs[i].set_xlabel('Epochs')
#     axs[i].set_ylabel(plot_labels[i])
#     axs[i].legend()

# # Adjust layout and display the plot
# plt.tight_layout()
# plt.show()