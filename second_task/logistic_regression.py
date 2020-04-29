import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten

from second_task.normalized_data import get_normalized_data, BATCH_SIZE, EPOCHS, NUM_CLASSES


# preparing data
(train_images, train_labels), (test_images, test_labels) = get_normalized_data()


# Defining the model
model = Sequential()

# Layer0 simply flattens image input
model.add(Flatten(input_shape=np.shape(train_images[0]), name='Images'))

# Layer1 is the output layer
model.add(Dense(units=NUM_CLASSES, activation=tf.nn.softmax, use_bias=True, name='Logistic'))


# Compile model
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

# Train model
model.fit(x=train_images, y=train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, validation_split=0.0)

# Evaluate model
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Prediction error
test_predictions = model.predict(test_images)
id_count = 0
n_errors = 0
for prediction in test_predictions:
    predicted_label = np.argmax(prediction)
    if predicted_label != np.argmax(test_labels[id_count]):
        n_errors += 1
    id_count += 1

error_rate = n_errors / float(np.shape(test_images)[0])
print(f"Prediction error is: {error_rate}")
