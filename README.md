# Micrograd

A minimal, educational implementation of automatic differentiation (autograd) engine and neural network library. This project was created to implement and understand backpropagation from scratch, providing insight into how modern deep learning frameworks work under the hood.


## Acknowledgments

- Educational approach based on Andrej Karpathy's micrograd tutorial - https://www.youtube.com/watch?v=VMj-3S1tku0
- Built for learning and understanding automatic differentiation

## Overview

Micrograd is a tiny scalar-valued autograd engine that implements backpropagation over a dynamically built DAG (Directed Acyclic Graph). It's designed to be simple, understandable, and educational, making it perfect for learning how automatic differentiation works in deep learning.

### What is Automatic Differentiation (Autograd)?

Automatic differentiation is a technique for efficiently computing derivatives of functions expressed as computer programs. In the context of machine learning, it's the backbone of backpropagation - the algorithm that enables neural networks to learn by computing gradients of loss functions with respect to model parameters.

Key features of this implementation:
- **Scalar-valued operations**: Works with individual numbers rather than tensors
- **Dynamic computation graph**: The graph is built on-the-fly as operations are performed
- **Reverse-mode autodiff**: Implements backpropagation efficiently
- **Educational focus**: Clean, readable code that's easy to understand

### Core Components

1. **Value Class**: The heart of the autograd engine
   - Wraps scalar values and tracks operations
   - Maintains gradients and computation graph structure
   - Supports basic arithmetic operations (+, -, *, /, **)
   - Includes activation functions (tanh, exp)

2. **Backpropagation**: Automatic gradient computation
   - Topological sorting of computation graph
   - Reverse-mode differentiation
   - Chain rule application

## Features

- ✅ Basic arithmetic operations with gradient tracking
- ✅ Common activation functions (tanh, exponential)
- ✅ Automatic backpropagation through computation graphs
- ✅ Clean, readable implementation perfect for learning
- ✅ Jupyter notebook examples and visualizations

## Installation and Setup

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Clone and Run Locally

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd micrograd
   ```

2. **Set up the environment using uv (recommended):**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create and activate virtual environment with dependencies
   uv sync
   ```

   **Or using pip:**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   
   # Install dependencies
   pip install -e .
   ```

3. **Explore the examples:**
   ```bash
   # Start Jupyter notebook to explore examples
   jupyter notebook playground.ipynb
   ```

## Usage Examples

### Basic Operations

```python
from micrograd.value import Value

# Create values
a = Value(2.0, _label='a')
b = Value(-3.0, _label='b')
c = Value(10.0, _label='c')

# Build computation graph
d = a * b
e = d + c
f = e.tanh()

# Perform backpropagation
f.backward()

# Check gradients
print(f"a.grad = {a.grad}")  # Gradient of f with respect to a
print(f"b.grad = {b.grad}")  # Gradient of f with respect to b
print(f"c.grad = {c.grad}")  # Gradient of f with respect to c
```

### Building a Simple Neural Network

```python
# Example of a simple neuron
x1, x2 = Value(1.0), Value(2.0)  # inputs
w1, w2 = Value(-3.0), Value(1.0)  # weights
b = Value(6.88137)  # bias

# Forward pass
z = x1*w1 + x2*w2 + b
o = z.tanh()

# Backward pass
o.backward()

print(f"Output: {o.data}")
print(f"Gradients: w1={w1.grad}, w2={w2.grad}, b={b.grad}")
```

## Project Structure

```
micrograd/
├── micrograd/
│   ├── __init__.py          # Package initialization
│   └── value.py             # Core Value class with autograd
├── playground.ipynb         # Jupyter notebook with examples
├── pyproject.toml          # Project configuration
├── README.md               # This file
└── uv.lock                 # Dependency lock file
```

## Learning Resources

This implementation is inspired by and follows the educational approach from:

**🎥 [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)** by Andrej Karpathy

This excellent video tutorial walks through building a similar autograd engine from scratch, explaining the mathematical foundations and implementation details step by step.

## Understanding the Code

### The Value Class

The `Value` class is the core of the autograd engine. Each instance represents a node in the computation graph:

- `data`: The actual numerical value
- `grad`: The gradient (derivative) of the final output with respect to this value
- `_prev`: Set of parent nodes in the computation graph
- `_op`: String representation of the operation that created this value
- `_backward`: Function that computes gradients for this operation

### Backpropagation Process

1. **Forward Pass**: Operations build a computation graph automatically
2. **Topological Sort**: Orders nodes to ensure gradients flow correctly
3. **Backward Pass**: Applies chain rule to compute gradients recursively

## Contributing

This is an educational project! Feel free to:
- Add more operations (sin, cos, log, etc.)
- Implement additional activation functions
- Add visualization tools for computation graphs
- Create more examples and tutorials

## License

This project is open source and available under the MIT License.
