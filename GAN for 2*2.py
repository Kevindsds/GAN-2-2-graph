# import packages
import numpy as np
from numpy import random
from matplotlib import pyplot as plt


# Define a function to produce a 2*2 graph according to data
def sample_visual(sample, m, n):
    fig, axes = plt.subplots(figsize=(10, 10), nrows=m, ncols=n, sharey=True,
                             sharex=True)
    for ax, img in zip(axes.flatten(), sample):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(1 - img.reshape((2, 2)), cmap='Greys_r')
    plt.show()
    return fig, axes


# Example of Real Photos
faces = [np.array([0, 1, 1, 0]),
         np.array([0.1, 0.9, 0.9, 0.1]),
         np.array([0.2, 0.9, 0.8, 0.1]),
         np.array([0, 0.8, 0.9, 0.2])
         ]

# a = sample_visual(faces, 1, 4)
# print(a)


# Build a sigmoid function
def sigm(x):
    return np.exp(x) / (1 + np.exp(x))


# Discriminator class
class Discriminator:
    def __init__(self):
        """Random pick a vector with four elements as a weight.
        Random pick a value as bias. """
        self.weight = np.array([np.random.normal() for i in range(4)])
        self.bias = np.random.normal()

    def forward(self, x):
        """Build a function to calculate prediction of discriminator. """
        return sigm(np.dot(x, self.weight) + self.bias)

    def real_error(self, image):
        """The value of prediction for real photo need to approach 1, so I use
        the formula -np.log(prediction). The closer the prediction is to 1, the
        smaller the error."""
        a = self.forward(image)
        return -np.log(a)

    def real_derivative(self, image):
        """Weight and bias approach the most appropriate value in the most
        efficient way for real photo."""
        a = self.forward(image)
        der_weight = - image * (1 - a)
        der_bias = - (1 - a)
        return der_weight, der_bias

    def real_update(self, image):
        """Update the value of weight and bias for real photo according to the
        derivative."""
        b = self.real_derivative(image)
        self.weight -= b[0]
        self.bias -= b[1]

    def fake_error(self, y):
        """The value of prediction for fake photo need to approach 0, so I use
        the formula -np.log(1-prediction). The closer the prediction is to 1,
        the smaller the error."""
        c = self.forward(y)
        return -np.log(1 - c)

    def fake_derivative(self, y):
        """Weight and bias approach the most appropriate value in the most
        efficient way for fake photo."""
        c = self.forward(y)
        der_weight = c * y
        der_bias = c
        return der_weight, der_bias

    def fake_update(self, y):
        """Update the value of weight and bias for fake photo according to the
        derivative."""
        d = self.fake_derivative(y)
        self.weight -= d[0]
        self.bias -= d[1]


# Generator Class
class Generator:
    def __init__(self):
        """Random pick a vector with four elements as a weight.
        Random pick a vector with four elements as a bias. """
        self.weight = np.array([np.random.normal() for i in range(4)])
        self.bias = np.array([np.random.normal() for i in range(4)])

    def forward(self, z):
        """Calculate the Predictions of the Generator according to the value of
        z. """
        return sigm(z * self.weight + self.bias)

    def error(self, z, discriminator):
        """As a Generator, the value of prediction for fake photo need to
        approach 1, so I use the formula -np.log(prediction). The closer the
        prediction is to 1, the smaller the error."""
        a = self.forward(z)
        b = discriminator.forward(a)
        return -np.log(b)

    def derivative(self, z, discriminator):
        """Weight and bias approach the most appropriate value in the most
        efficient way to produce photos that closely resemble real photos . """
        discriminator_weight = discriminator.weight
        discriminator_bias = discriminator.bias
        x = self.forward(z)
        y = discriminator.forward(x)
        factor = -(1 - y) * discriminator_weight * x * (1 - x)
        derivatives_weights = factor * z
        derivative_bias = factor
        return derivatives_weights, derivative_bias

    def update(self, z, discriminator, learning_rate):
        """"Update the value of weight and bias according to the derivative."""
        c = self.derivative(z, discriminator)
        self.weight -= learning_rate * c[0]
        self.bias -= learning_rate * c[1]


# Prepare for Training code
np.random.seed(30)
learning_rate = 0.01
epochs = 1000
D = Discriminator()
G = Generator()


# Training code
for epoch in range(epochs):
    for face in faces:
        D.real_update(face)
        z = np.random.randn()
        G.forward(z)
        D.fake_update(z)
        G.update(z, D, learning_rate)

# Application
generated_images = []
for i in range(4):
    z = random.random()
    generated_image = G.forward(z)
    generated_images.append(generated_image)

# Photos produced by the Generator
print(sample_visual(generated_images, 1, 4))
for i in generated_images:
    print(i)

# Value of weight and bias for the Generator and the Discriminator after
# training
print("Generator weights", G.weight)
print("Generator biases", G.bias)
print("Discriminator weights", D.weight)
print("Discriminator bias", D.bias)
