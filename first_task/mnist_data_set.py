from first_task.neural_network import NeuralNetwork
from first_task.layer import Layer
from first_task.activation_layer import ActivationLayer
from first_task.activation_funcs import tanh, tanh_prime
from first_task.loss_funcs import mse, mse_prime

from keras.datasets import mnist
from keras.utils import np_utils

# loading MNIST data set from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
x_train = x_train.astype('float32')
x_train = x_train / 255

# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data: 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
x_test = x_test.astype('float32')
x_test = x_test / 255
y_test = np_utils.to_categorical(y_test)

# Network
net = NeuralNetwork()
net.add(Layer(28 * 28, 100))  # input_shape=(1, 28 * 28); output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(Layer(100, 50))  # input_shape=(1, 100); output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(Layer(50, 10))  # input_shape=(1, 50); output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))


# train on 1000 samples
net.use(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

# test on 3 samples
print(f"\nPredicted values : {net.predict(x_test[0:3])}\n")
print(f"true values : {y_test[0:3]}")
