"""
This module implements a simple neural network framework with basic components like Neurons, Layers, and Multi-Layer Perceptrons (MLP).
It provides a structure for building and training neural networks using custom Value objects for automatic differentiation.
"""

import numpy as np
from micrograd.value import Value


class Module:
    """Base class for all neural network modules."""
    def __init__(self):
        pass

    def parameters(self):
        return []

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implemented")

    def backward(self, *args, **kwargs):
        raise NotImplementedError("Backward method not implemented")

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0

class Neuron(Module):
    """A simple neuron that computes a weighted sum of inputs and adds a bias."""

    def __init__(self, input_size):
        super().__init__()
        self.weights = [Value(np.random.uniform(-1, 1)) for _ in range(input_size)]
        self.bias = Value(np.random.uniform(-1, 1))

    def forward(self, inputs):
        assert len(inputs) == len(self.weights), "Input size must match weights size"
        z = sum((w * x for w, x in zip(self.weights, inputs)), start=self.bias)
        out = z.tanh()  # Using tanh activation function
        return out

    def parameters(self):
        return self.weights + [self.bias]

class Layer(Module):
    """A layer of neurons."""
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def forward(self, inputs):
        out =  [neuron.forward(inputs) for neuron in self.neurons]
        return out if len(out) > 1 else out[0]


    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

class MLP(Module):
    """A simple multi-layer perceptron."""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        size = [input_size] + hidden_sizes + [output_size]
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(size) - 1)]

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]