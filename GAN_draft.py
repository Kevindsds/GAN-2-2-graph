# import packages
import numpy as np
from numpy import random
from matplotlib import pyplot as plt


# Draw function
def sample_visual(sample, m, n):
    fig, axes = plt.subplots(figsize=(10, 10), nrows=m, ncols=n, sharey=True,
                             sharex=True)
    for ax, img in zip(axes.flatten(), sample):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(1 - img.reshape((2, 2)), cmap='Greys_r')
    return fig, axes


# import real photos (can't view the graph)
faces = [np.array([0, 1, 1, 0]),
         np.array([0.1, 0.9, 0.9, 0.1]),
         np.array([0.2, 0.9, 0.7, 0.1]),
         np.array([0, 0.7, 0.9, 0.2])
         ]


# a = sample_visual(faces, 1, 4)
# print(a)

# sigmoid function
def sigm(x):
    return np.exp(x) / (1 + np.exp(x))


# Discriminator class
class Discriminator:
    def __init__(self):
        self.weight = np.array([np.random.normal() for i in range(4)])
        self.bias = np.random.normal()

    def forward(self, x):
        return sigm(np.dot(x, self.weight) + self.bias)

    def real_error(self, image):
        a = self.forward(image)
        return -np.log(a)

    def real_derivative(self, image):
        a = self.forward(image)
        der_weight = - image * (1 - a)
        der_bias = - (1 - a)
        return der_weight, der_bias

    def real_update(self, image):
        b = self.real_derivative(image)
        self.weight -= b[0]
        self.bias -= b[1]

    def fake_error(self, y):
        c = self.forward(y)
        return -np.log(1 - c)

    def fake_derivative(self, y):
        c = self.forward(y)
        der_weight = c * y
        der_bias = c
        return der_weight, der_bias

    def fake_update(self, y):
        d = self.fake_derivative(y)
        self.weight -= d[0]
        self.bias -= d[1]


# Generator Class
class Generator:
    def __init__(self):
        self.weight = np.array([np.random.normal() for i in range(4)])
        self.bias = np.array([np.random.normal() for i in range(4)])

    def forward(self, z):
        return sigm(z * self.weight + self.bias)

    def error(self, z, discriminator):
        a = self.forward(z)
        b = discriminator.forward(a)
        return -np.log(b)

    def derivative(self, z, discriminator):
        discriminator_weight = discriminator.weight
        discriminator_bias = discriminator.bias
        x = self.forward(z)
        y = discriminator.forward(x)
        factor = -(1 - y) * discriminator_weight * x * (1 - x)
        derivatives_weights = factor * z
        derivative_bias = factor
        return derivatives_weights, derivative_bias

    def update(self, z, discriminator, learning_rate):
        c = self.derivative(z, discriminator)
        self.weight -= learning_rate * c[0]
        self.bias -= learning_rate * c[1]


# prepare for Training code
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
_ = sample_visual(generated_images, 1, 4)
for i in generated_images:
    print(i)

# Value of weight and bias
print("Generator weights", G.weight)
print("Generator biases", G.bias)
print("Discriminator weights", D.weight)
print("Discriminator bias", D.bias)
