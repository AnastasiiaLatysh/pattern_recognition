
import numpy as np

from first_task.neural_network import NeuralNetwork
from first_task.layer import Layer
from first_task.activation_layer import ActivationLayer
from first_task.activation_funcs import tanh, tanh_prime
from first_task.loss_funcs import mse, mse_prime

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = NeuralNetwork()
net.add(Layer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(Layer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
