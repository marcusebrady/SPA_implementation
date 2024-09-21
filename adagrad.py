import numpy as np
# use this paper for implementation https://arxiv.org/pdf/2405.01908

class Adagrad:
    def __init__(self, params, lr=0.01, eps=1e-8):
        self.params = params
        self.lr = lr
        self.eps = eps
        self.G = []
        for p in self.params:
            if isinstance(p.value, np.ndarray):
                self.G.append(np.zeros_like(p.value))
            elif isinstance(p.value, (int, float)):
                self.G.append(0.0)
            else:
                raise TypeError("Unsupported parameter type for Adagrad optimizer.")

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                grad_val = param.grad.value  
                if isinstance(self.G[i], np.ndarray):
                    self.G[i] += grad_val ** 2

                    param.value -= self.lr * grad_val / (np.sqrt(self.G[i]) + self.eps)
                elif isinstance(self.G[i], float):
                    self.G[i] += grad_val ** 2
                    param.value -= self.lr * grad_val / (np.sqrt(self.G[i]) + self.eps)

    def zero_grad(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                if isinstance(self.G[i], np.ndarray):
                    param.grad.value = 0.0  
                elif isinstance(self.G[i], float):
                    param.grad.value = 0.0


