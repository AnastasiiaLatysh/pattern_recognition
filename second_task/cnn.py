import tensorflow as tf
import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

from second_task.normalized_data import get_normalized_data, NUM_CLASSES, BATCH_SIZE, EPOCHS


# data preparation
(train_images, train_labels), (test_images, test_labels) = get_normalized_data()

# Must define the input shape in the first layer of the neural network
input_shape = (train_images.shape[1:] + (1,))  # (28, 28, 1)

# Defining the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

inp = Input(shape=input_shape)
_ = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inp)
_ = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(_)
_ = MaxPool2D(pool_size=(2, 2))(_)
_ = Dropout(0.25)(_)
_ = Flatten()(_)
_ = Dense(units=128, activation='relu')(_)
_ = Dropout(0.2)(_)
_ = Dense(units=NUM_CLASSES, activation='softmax')(_)
model = Model(inputs=inp, outputs=_)
model.summary()

# Compile model
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

# Train model
history = model.fit(np.expand_dims(train_images, -1), train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_split=0.0)

score = model.evaluate(np.expand_dims(test_images, -1), test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Prediction error
test_predictions = model.predict(np.expand_dims(test_images, -1))
id_count = 0
n_errors = 0
for prediction in test_predictions:
    predicted_label = np.argmax(prediction)
    if predicted_label != np.argmax(test_labels[id_count]):
        n_errors += 1
    id_count += 1

error_rate = n_errors / float(np.shape(test_images)[0])
print(f"Prediction error is: {error_rate}")
