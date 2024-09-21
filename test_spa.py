import numpy as np
from spa import compute_spa
import sys
import matplotlib.pyplot as plt

# i like gruvbox theme 
gruvbox_bg = "#282828"
gruvbox_fg = "#ebdbb2"
gruvbox_yellow = "#fabd2f"
gruvbox_red = "#fb4934"
gruvbox_blue = "#83a598"
gruvbox_green = "#b8bb26"
gruvbox_orange = "#fe8019"
gruvbox_purple = "#d3869b"

def plot_function_and_integrand(f, zeta, x_range=(-2, 4), num_points=1000):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = f(x)
    integrand = np.exp(-zeta * y)
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(12, 8))  
    ax.plot(x, integrand, label=r'$e^{-\zeta f(x)}$', color=gruvbox_blue, linewidth=2)
    ax.axvline(x=1.0, color=gruvbox_red, linestyle='--', linewidth=1.5, label='Minimum $x_0=1$')
    ax.set_yscale('log')
    ax.set_title(f'Integrand $e^{{-\zeta f(x)}}$ for zeta = {zeta} (Log Scale)', color=gruvbox_fg, fontsize=16, pad=15)
    ax.set_xlabel('x', color=gruvbox_fg, fontsize=14, labelpad=10)
    ax.set_ylabel(r'$e^{-\zeta f(x)}$', color=gruvbox_fg, fontsize=14, labelpad=10)
    ax.legend(loc='upper right', fontsize=12, frameon=False)

    ax.tick_params(axis='x', colors=gruvbox_fg, labelsize=12)
    ax.tick_params(axis='y', colors=gruvbox_fg, labelsize=12)
    ax.spines['top'].set_color(gruvbox_fg)
    ax.spines['right'].set_color(gruvbox_fg)
    ax.spines['left'].set_color(gruvbox_fg)
    ax.spines['bottom'].set_color(gruvbox_fg)
    ax.grid(True, color=gruvbox_fg, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_facecolor(gruvbox_bg)
    fig.patch.set_facecolor(gruvbox_bg)
    plt.tight_layout()
    plt.savefig(f'./{zeta}_integrand.png', dpi=300, transparent=True)
    plt.close()  



def quartic_function(params, X, Y):
    """
    quartic function: f(x) = (x - 1)^4 + 12*(x - 1)^2 + 1
        f(1) = 0 + 0 + 1 = 1
        f'(1) = 0
        f''(1) = 24
        f'''(1) = 0
        f''''(1) = 24
    """
    x = params[0]
    return (x - 1)**4 + 12*(x - 1)**2 + 1

def test_spa(zeta_values, x_initial=0.0, num_steps=10000, tol=1e-6):
    for zeta in zeta_values:
        print(f"\n=== Testing SPA with zeta = {zeta} ===")
        if zeta == 500:
            plot_function_and_integrand(lambda x: (x - 1)**4 + 12*(x - 1)**2 + 1, zeta)

        try:
            spa_results = compute_spa(
                f=quartic_function,
                x_init=x_initial,
                zeta=zeta
            )
            print("\n--- Saddle Point Approximation Results ---")
            print(f"Approximated Integral (J_spa): {spa_results['J_spa']}")
            print(f"Exact Integral for Quadratic (J_exact): {spa_results['J_exact']}")
            print(f"Numerical Integral (J_numeric): {spa_results['J_numeric']}")
            print(f"Minimum Position (x0): {spa_results['x0']}")
            print(f"Function Value at Minimum (f0): {spa_results['f0']}")
            print(f"Second Derivative at x0 (f''): {spa_results['f_double_prime']}")
            print(f"Third Derivative at x0 (f'''): {spa_results['f_triple_prime']}")
            print(f"Fourth Derivative at x0 (f''''): {spa_results['f_quadruple_prime']}")


            expected_x0 = 1.0
            expected_f0 = 1.0
            expected_f_double_prime = 24.0
            expected_f_triple_prime = 0.0
            expected_f_quadruple_prime = 24.0

            assert np.isclose(spa_results['x0'], expected_x0, atol=1e-5), \
                f"x0 expected: {expected_x0}, got: {spa_results['x0']}"
            assert np.isclose(spa_results['f0'], expected_f0, atol=1e-5), \
                f"f0 expected: {expected_f0}, got: {spa_results['f0']}"
            assert np.isclose(spa_results['f_double_prime'], expected_f_double_prime, atol=1e-5), \
                f"f'' expected: {expected_f_double_prime}, got: {spa_results['f_double_prime']}"
            assert np.isclose(spa_results['f_triple_prime'], expected_f_triple_prime, atol=1e-4), \
                f"f''' expected: {expected_f_triple_prime}, got: {spa_results['f_triple_prime']}"
            assert np.isclose(spa_results['f_quadruple_prime'], expected_f_quadruple_prime, atol=1e-5), \
                f"f'''' expected: {expected_f_quadruple_prime}, got: {spa_results['f_quadruple_prime']}"

            relative_error = abs(spa_results['J_spa'] - spa_results['J_exact']) / spa_results['J_exact']
            print(f"Relative Error between J_spa and J_exact: {relative_error:.6f}")
            assert relative_error < 1e-2, \
                f"Relative error too high: {relative_error:.6f}"

            print("Test passed successfully.")

        except AssertionError as ae:
            print(f"Assertion Error: {ae}")
        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)

def main():
    zeta_values = [10, 50, 100, 200, 500]
    test_spa(zeta_values=zeta_values)

if __name__ == "__main__":
    main()


