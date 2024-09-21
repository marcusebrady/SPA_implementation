import numpy as np
from autograd_engine import Value
from adagrad import Adagrad

def find_global_minimum(f, x_init, lr=0.1, num_steps=10000, tol=1e-6):
    """
    f (callable): the function to minimize.
    x_init (float): initial guess for x.

    returns:
    tuple: (x0, f0) where x0 is the position of the minimum and f0 is the function value at x0.
    """
    x = Value(x_init, requires_grad=True)
    params = [x]
    optimizer = Adagrad(params, lr=lr)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        loss = f(params, None, None)  
        loss.backward()
        optimizer.step()

        grad_norm = np.linalg.norm([p.grad.value for p in params if p.grad is not None])
        if grad_norm < tol:
            print(f"Convergence reached at step {step}.")
            break
        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss.value}, Gradient Norm: {grad_norm}")

    x0 = x.value
    f0 = f(params, None, None).value

    return x0, f0

def compute_derivatives(f, x0):
    x = Value(x0, requires_grad=True)
    params = [x]
    f_value = f(params, None, None)
    f_prime = f_value.derivative([x], create_graph=True)[0] 
    f_double_prime = f_prime.derivative([x], create_graph=True)[0]  
    f_triple_prime = f_double_prime.derivative([x], create_graph=True)[0]
    f_quadruple_prime = f_triple_prime.derivative([x], create_graph=False)[0] 
    return f_double_prime.value, f_triple_prime.value, f_quadruple_prime

def saddle_point_approximation(f0, f_double_prime, f_quadruple_prime, zeta):
    if f_double_prime <= 0:
        raise ValueError("Second derivative must be positive for SPA.")
    J0 = np.sqrt(2 * np.pi / (zeta * f_double_prime)) * np.exp(-zeta * f0)
    correction = 1 - (3 * zeta * f_quadruple_prime) / (4 * (zeta * f_double_prime)**2)
    J_spa = J0 * correction
    return J_spa

def exact_integral_quadratic(f0, f_double_prime, zeta):
    return np.sqrt(2 * np.pi / (zeta * f_double_prime)) * np.exp(-zeta * f0)

def numerical_integration(f, zeta, x_range, num_points=100000):
    """
    simply for testing purposes
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    dx = (x_range[1] - x_range[0]) / (num_points - 1)
    integrand = np.exp(-zeta * f(x))
    return np.sum(integrand) * dx

def compute_spa(f, x_init, zeta):
    """
    returns a dictionary containing the results
    """

    x0, f0 = find_global_minimum(f, x_init)
    f_double_prime, f_triple_prime, f_quadruple_prime = compute_derivatives(f, x0)
    J_spa = saddle_point_approximation(f0, f_double_prime, f_quadruple_prime, zeta)
    J_exact = exact_integral_quadratic(f0, f_double_prime, zeta)

    def f_numeric(x):
        return (x - 1)**4 + 12*(x - 1)**2 + 1

    x_range = (x0 - 5, x0 + 5)  
    J_numeric = numerical_integration(f_numeric, zeta, x_range)

    return {
        'J_spa': J_spa,
        'J_exact': J_exact,
        'J_numeric': J_numeric,
        'x0': x0,
        'f0': f0,
        'f_double_prime': f_double_prime,
        'f_triple_prime': f_triple_prime,
        'f_quadruple_prime': f_quadruple_prime
    }


