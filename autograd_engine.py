import numpy as np

class Value:
    def __init__(self, value, _children=(), _op='', requires_grad=True):
        self.value = float(value) if isinstance(value, (int, float)) else value
        self.requires_grad = requires_grad
        self.grad = Value(0.0, requires_grad=False) if requires_grad else None
        self._prev = set(_children)
        self._op = _op
        self._backwards = lambda: None
        self._backward_hooks = []

    def __repr__(self):
        return f"Value(value={self.value}, grad={self.grad.value if self.grad else None})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, (self, other), '+')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (out.grad if out.grad else Value(0.0))
            if other.requires_grad:
                other.grad = other.grad + (out.grad if out.grad else Value(0.0))

        out._backwards = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value * other.value, (self, other), '*')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (other * (out.grad if out.grad else Value(0.0)))
            if other.requires_grad:
                other.grad = other.grad + (self * (out.grad if out.grad else Value(0.0)))

        out._backwards = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Value(self.value ** other, (self,), f'**{other}')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (Value(other) * (self ** (other - 1)) * (out.grad if out.grad else Value(0.0)))

        out._backwards = _backward
        return out

    def __rmul__(self, other):          
        return self * other

    def __truediv__(self, other):  
        return self * (other ** -1)

    def __rtruediv__(self, other):  
        return other * (self ** -1)

    def __neg__(self):  
        return self * -1

    def __sub__(self, other):  
        return self + (-other)

    def __rsub__(self, other):  
        return other + (-self)

    def exp(self):
        out = Value(np.exp(self.value), (self,), 'exp')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (out * (out.grad if out.grad else Value(0.0)))

        out._backwards = _backward
        return out

    def sin(self):
        out = Value(np.sin(self.value), (self,), 'sin')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (Value(np.cos(self.value)) * (out.grad if out.grad else Value(0.0)))

        out._backwards = _backward
        return out

    def cos(self):
        out = Value(np.cos(self.value), (self,), 'cos')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (Value(-np.sin(self.value)) * (out.grad if out.grad else Value(0.0)))

        out._backwards = _backward
        return out

    def backward(self, gradient=None):
        if not self.requires_grad:
            return

        if gradient is None:
            gradient = Value(1.0, requires_grad=False)

        self.grad = self.grad + gradient

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in reversed(topo):
            node._backwards()

    def zero_grad(self):
        if self.grad is not None:
            self.grad = Value(0.0, requires_grad=False)
        for child in self._prev:
            if isinstance(child, Value):
                child.zero_grad()

    def derivative(self, vars, create_graph=False):
        if not self.requires_grad:
            raise RuntimeError("Can't compute derivatives for objects that do not require gradients.")

        if not isinstance(vars, list):
            raise TypeError("Variables must be provided as a list of Value objects.")
        self.zero_grad()
        self.backward(Value(1.0, requires_grad=False))
        grads = []
        for var in vars:
            if var.grad is not None:
                if create_graph:
                    grad = var.grad
                    grad.requires_grad = True                  
                else:
                    grad = var.grad.value
                grads.append(grad)
            else:
                grads.append(0.0)
        if not create_graph:
            self.zero_grad()

        return grads


