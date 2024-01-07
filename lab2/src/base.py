import numpy as np
import matplotlib.pyplot as plt
import json

from typing import Callable
from scipy.optimize import minimize
import sys

with open('configs/constants.json') as file:
    data = json.load(file)

C, T, G, T0, N_RANGE = data.values()


def composite_simpson(a: float, b: float, n: int, f: Callable) -> float:
    if a > b:
        raise ValueError('a > b')
    if n < 3:
        raise ValueError('n < 3')
    if not isinstance(f, Callable):
        raise TypeError('f is not callable')

    if n % 2 == 0:
        n -= 1

    h = (b - a) / (n - 1)

    x_values = np.linspace(a, b, n)
    f_values = f(x_values)

    odd_sum = np.sum(f_values[2:-1:2])
    even_sum = np.sum(f_values[1:-1:2])

    return (h / 3) * (f_values[0] + 2 * odd_sum +
                      4 * even_sum + f_values[-1])


def composite_trapezoid(a: float, b: float, n: int, f: Callable) -> float:
    if a > b:
        raise ValueError('a > b')
    if n < 2:
        raise ValueError('n < 2')
    if not isinstance(f, Callable):
        raise TypeError('f is not callable')

    h = (b - a) / (n - 1)
    x_values = np.linspace(a, b, n)
    f_values = f(x_values)

    sum = np.sum(f_values[1:-1])

    return (h / 2) * (f_values[0] + 2 * sum + f_values[-1])


def func_for_integration(t: np.ndarray, option: int = None) -> np.ndarray:
    def y(t): return C * (1/2) * (1 - np.cos(2*t))
    def dxdt(t): return C * (1 - np.cos(2*t))
    def dydt(t): return C * np.sin(2*t)
    def dydx(t): return dydt(t) / dxdt(t)

    # Вычисляем значения функции для всех t

    variants = {
        1: np.sqrt(2*C/G),
        2: 0
    }

    # Возвращаем специальное значение, если t равно нулю, и обычное значение в противном случае
    return np.where(t == 0, variants.get(option), np.sqrt((1 + dydx(t) ** 2) / (2 * G * y(t))) * dxdt(t))


def get_real_value(T0: float) -> float:
    return np.sqrt(2 * C / G) * (T - T0)


def integrate(integrant: Callable, T0: float) -> None:
    N = np.arange(N_RANGE['min'], N_RANGE['max'], N_RANGE['step'])

    h = (T - T0) / (N - 1)
    simpson = [composite_simpson(T0, T, n, integrant) for n in N]
    trapezoid = [composite_trapezoid(
        T0, T, n, integrant) for n in N]

    return h, simpson, trapezoid


def calculate_absolute_error(real: float, approx: float) -> float:
    return np.absolute(real - approx)


def plot_orders_of_accuracy(h: np.ndarray) -> None:
    plt.plot(h, h, label=r'$O(h)$', color='#696969', zorder=1, linestyle='-')
    plt.plot(h, h ** 2, label=r'$O(h^2)$',
             color='#696969', zorder=1, linestyle='--')
    plt.plot(h, h ** 4, label=r'$O(h^4)$',
             color='#696969', zorder=1, linestyle='-.')


def lab_base():
    h, simpson, trapezoid = integrate(
        lambda t: func_for_integration(t, option=2), 0)
    real = get_real_value(0)

    simpson_error = calculate_absolute_error(real, simpson)
    trapezoid_error = calculate_absolute_error(real, trapezoid)

    plt.figure(figsize=(7, 7))
    plt.scatter(h, simpson_error, label='Simpson Error',
                s=10, color='#DC143C', zorder=2)
    plt.scatter(h, trapezoid_error,
                label='Trapezoid Error', s=10, color='#1E90FF', zorder=2)
    plot_orders_of_accuracy(h)
    # plt.plot(h, [sys.float_info.epsilon for _ in h],
    #          lw=2, color='orange', zorder=0, label=r'$\epsilon$')
    plt.xlabel('Шаг интегрирования')
    plt.ylabel('Абсолютная погрешность')
    plt.loglog()
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_integral_values():
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    N = np.arange(N_RANGE['min'], N_RANGE['max'], N_RANGE['step'])
    h, simpson, trapezoid = integrate(func_for_integration, T0)
    real = [get_real_value(T0) for _ in N]

    ax[0].plot(N, simpson, label='Simpson', color='#DC143C', zorder=2, lw=3)
    ax[1].plot(N, trapezoid, label='Trapezoid',
               color='#1E90FF', zorder=2, lw=3)

    ax[0].set_xlabel('Количество узлов интегрирования')
    ax[0].set_ylabel('Значение интеграла')
    ax[0].legend()
    ax[0].grid()

    ax[1].set_xlabel('Количество узлов интегрирования')
    ax[1].set_ylabel('Значение интеграла')
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.show()


lab_base()
