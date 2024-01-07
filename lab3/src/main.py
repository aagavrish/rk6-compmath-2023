import os
import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from scipy.optimize import root
from typing import Callable, List, Tuple


def f(t: float, theta: np.ndarray) -> np.ndarray:
    theta1, theta2 = theta
    return np.array([theta2, np.cos(t) - 0.1*theta2 - np.sin(theta1)])


def runge_kutta(x_0: np.ndarray, t_n: float,
                f: Callable, h: float) -> Tuple[np.ndarray, np.ndarray]:
    t_values = np.arange(0, t_n + h, h)
    y_values = np.zeros((len(t_values), len(x_0)))

    y_values[0] = x_0
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]

        k1 = h * f(t, y)
        k2 = h * f(t + 0.5*h, y + 0.5*k1)
        k3 = h * f(t + 0.5*h, y + 0.5*k2)
        k4 = h * f(t + h, y + k3)

        y_values[i] = y + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_values, y_values


def adams_moulton(x_0: np.ndarray, t_n: float,
                  f: Callable, h: float) -> Tuple[np.ndarray, np.ndarray]:
    t_values = np.arange(0, t_n + h, h)
    y_values = np.zeros((len(t_values), len(x_0)))

    _, y_rk_values = runge_kutta(x_0, 2*h, f, h)
    for j in range(3):
        y_values[j] = y_rk_values[j]

    for i in range(3, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]
        y_prev = y_values[i - 2]
        y_prev2 = y_values[i - 3]

        def func(y_next):
            return y_next - (y + h/24 * (9*f(t + h, y_next) +
                                         19*f(t, y) - 5*f(t - h, y_prev) +
                                         f(t - 2*h, y_prev2)))

        sol = root(func, y)
        y_values[i] = sol.x

    return t_values, y_values


def milne_simpson(x_0: np.ndarray, t_n: float,
                  f: Callable, h: float) -> Tuple[np.ndarray, np.ndarray]:
    t_values = np.arange(0, t_n + h, h)
    y_values = np.zeros((len(t_values), len(x_0)))

    _, y_rk_values = runge_kutta(x_0, 3*h, f, h)
    for j in range(4):
        y_values[j] = y_rk_values[j]

    for i in range(4, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]
        y_prev = y_values[i - 2]
        y_prev2 = y_values[i - 3]
        y_prev3 = y_values[i - 4]

        y_pred = y_prev3 + 4*h/3 * \
            (2*f(t, y) - f(t - h, y_prev) +
             2*f(t - 2*h, y_prev2))

        y_next = y_prev + h/3 * \
            (f(t + h, y_pred) + 4*f(t, y) +
             f(t - h, y_prev))

        y_values[i] = y_next

    return t_values, y_values


def plot(h: float, t_n: float, method: Callable, method_name: str, theta_2: np.ndarray) -> None:
    theta_1 = 0

    start = timer()
    results = [(theta, method([theta_1, theta], t_n, f, h))
               for theta in theta_2]
    end = timer()
    print(f'{method_name} method took {end - start:.2f}s')

    def plot_and_save(title: str, xlabel: str, ylabel: str,
                      filename: str, plot_data: List) -> None:
        plt.figure(figsize=(14, 4))
        for theta, data in plot_data:
            plt.plot(*data, label=f'{theta:.2f}')
        # plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(f'images/{filename}.pdf')
        # plt.show()

    plot_and_save(f'Trajectories of $\\theta(t)$ using {method_name} method',
                  '$t$', '$\\theta(t)$', f'{method_name}_theta_t',
                  [(theta, (t, y[:, 0]))
                   for theta, (t, y) in results])

    plot_and_save(f'Phase trajectories using {method_name} method',
                  '$\\theta$', '$d\\theta/dt$', f'{method_name}_phase',
                  [(theta, (y[:, 0], y[:, 1]))
                   for theta, (_, y) in results])


def plot_fixed(h: float, t_n: float,
               methods: List[Tuple[str, Callable]]) -> None:
    theta_1 = 0
    theta_2 = [2.0]

    results = []
    for method_name, method in methods:
        result = [(theta, method([theta_1, theta], t_n, f, h))
                  for theta in theta_2]
        results.extend([(method_name, theta, data) for theta, data in result])

    def plot_and_save(title: str, xlabel: str, ylabel: str,
                      filename: str, plot_data: List) -> None:
        plt.figure(figsize=(9, 5))
        for method_name, theta, data in plot_data:
            plt.plot(*data, label=f'{method_name}')
        # plt.title(f'{title} ($\\frac{{d\\theta}}{{dt}}$ = {theta:.2f})')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'images/{filename}.pdf')
        # plt.show()

    plot_and_save('Phase trajectories using different methods', '$\\theta$',
                  '$d\\theta/dt$', 'Phase-Trajectories',
                  [(method_name, theta, (y[:, 0], y[:, 1]))
                   for method_name, theta, (_, y) in results])


def plot_single(h: float, t_n: float, method: Callable, color: str, theta_2: float, method_name: str) -> None:
    theta_1 = 0

    start = timer()
    result = method([theta_1, theta_2], t_n, f, h)
    end = timer()
    print(f'{method_name} method took {end - start:.2f}s')

    t, y = result
    plt.figure(figsize=(14, 4))
    plt.plot(t, y[:, 0], color=color, label=f'{theta_2:.2f}')
    plt.xlabel('$t$')
    plt.ylabel('$\\theta(t)$')
    plt.tight_layout()

    directory = f'images/{method_name}-h'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f'{directory}/{method_name}_h={h}.pdf')
    # plt.show()


if __name__ == '__main__':
    h, t_n = 0.001, 300
    theta_2 = np.random.uniform(1.85, 2.1, 15)

    # plot(h, t_n, runge_kutta, 'Runge-Kutta', theta_2)
    # plot(h, t_n, adams_moulton, 'Adams-Moulton', theta_2)
    # plot(h, t_n, milne_simpson, 'Milne-Simpson', theta_2)

    # plot_fixed(h, t_n, [('Runge-Kutta', runge_kutta), ('Adams-Moulton',
    #            adams_moulton), ('Milne-Simpson', milne_simpson)])

    # h, t_n = np.arange(0.1, 0.25, 0.005), 200
    # theta_2 = 1.9

    # for h_current in h:
    #     plot_single(h_current, t_n, runge_kutta, 'red', theta_2, 'Runge-Kutta')
    #     plot_single(h_current, t_n, adams_moulton, 'green',
    #                 theta_2, 'Adams-Moulton')
    #     plot_single(h_current, t_n, milne_simpson,
    #                 'blue', theta_2, 'Milne-Simpson')
