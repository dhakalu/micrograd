import math

"""
A simple implementation of a value class that supports basic arithmetic operations
and keeps track of the operations performed on it for automatic differentiation.
"""
class Value:

    def __init__(self, data, _children=(), _op='', _label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self._label = _label

        self.grad = 0.0  # Initialize gradient to zero
        self._backward = lambda: None  # Placeholder for backward function

    def __repr__(self):
        return f"Value({self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Exponent must be a number"
        x = self.data 
        p = x ** other
        out = Value(p, (self,), f'**{other}')
        def _backward():
            self.grad += other * (x ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')
        def _backward():
            self.grad += out.grad / other.data
            other.grad -= self.data * out.grad / (other.data ** 2)
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        ex = math.exp(x)
        out = Value(ex, (self,), 'exp')
        def _backward():
            self.grad += ex * out.grad

        out._backward = _backward

        return out

    def backward(self):
        """
        Performs backpropagation to compute gradients.
        1. Initializes the gradient of the root node to 1.0.
        2. Builds a topological order of the computation graph.
        3. Iterates through the nodes in reverse topological order,
           calling their backward methods to propagate gradients.
        """
        self.grad = 1.0  # Initialize gradient for the root node
        topological_order = []
        visited = set()
        def build_topological_order(value):
            if value not in visited:
                visited.add(value)
                for child in value._prev:
                    build_topological_order(child)
                topological_order.append(value)
        build_topological_order(self)
        for value in reversed(topological_order):
            value._backward()