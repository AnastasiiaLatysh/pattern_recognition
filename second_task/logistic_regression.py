import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Download Fashion MNIST data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Training Parameters
batch_size = 128
n_epochs = 50

# Building a feed-forward network
model = tf.keras.Sequential()  # empty model

# Layer0 simply flattens image input
layer0 = tf.keras.layers.Flatten(input_shape=np.shape(train_images[0]), name='Images')
model.add(layer0)

# Layer1 is the output layer
num_units_l1 = 10
layer1 = tf.keras.layers.Dense(units=num_units_l1, activation=tf.nn.softmax, use_bias=True, name='logistic')
model.add(layer1)

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Plot Logistic Regression Network
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)


# Normalizing input
train_images = train_images / 255
test_images = test_images / 255

# Train model
train_history = model.fit(x=train_images, y=train_labels, batch_size=batch_size, epochs=n_epochs, shuffle=False,
                          validation_split=0.0)


plt.figure(1)
plt.plot(train_history.history['accuracy'], )

plt.figure(1)
plt.plot(train_history.history['accuracy'], 'b-o')
plt.grid(True)
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.savefig('Accuracy')

plt.figure(2)
plt.plot(train_history.history['loss'], 'r-s')
plt.grid(True)
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.savefig('Loss')
