import numpy as np
from keras import Sequential
from keras.datasets import cifar10
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.utils import np_utils


# Declare variables
batch_size = 32
num_classes = 10
epochs = 100

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert and pre-processing
x_train = x_train.reshape(x_train.shape[0], 3, 32, 32).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 3, 32, 32).astype('float32')
x_train = x_train / 255
x_test = x_test / 255
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


# Define model
model = Sequential()

# First hidden layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))

# Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening layer
model.add(Flatten())

# Fully connected layer
model.add(Dense(units=512, activation='relu'))

# Output layer
model.add(Dense(units=num_classes, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Fit model
cnn = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)


# Evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
