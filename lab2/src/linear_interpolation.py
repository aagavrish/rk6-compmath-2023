import numpy as np
from typing import List, Tuple
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from scipy.spatial import distance
import json
import time
from matplotlib import cm
import sys

with open('configs/constants.json') as file:
    data = json.load(file)

C, T, G, T0, N_RANGE = data.values()


def cycloid(t: float, C: float) -> Tuple[float, float]:
    return C * (t - 0.5 * np.sin(2 * t)), C * (0.5 - 0.5 * np.cos(2 * t))


def coeff_lin(y_nodes: np.ndarray, x_nodes: np.ndarray) -> List[Tuple[float, float]]:
    coeffs = []
    for i in range(len(x_nodes) - 1):
        a = (y_nodes[i + 1] - y_nodes[i]) / (x_nodes[i + 1] - x_nodes[i])
        b = y_nodes[i] - a * x_nodes[i]
        coeffs.append((a, b))  # Коэффициенты линейной функции y = ax + b
    return coeffs


def func(x: float, a: float, b: float) -> float:
    Y_j = a * x + b
    if Y_j <= 0:
        return 0
    Y_j_prime = a
    return np.sqrt((1 + Y_j_prime**2) / (2 * G * Y_j))


def composite_simpson(coeff: Tuple[float, float], func: callable,
                      a: float, b: float, n: int,
                      x_nodes: np.ndarray, y: np.ndarray) -> float:
    h = (b - a) / n
    integral = func(a, *coeff) + func(b, *coeff)
    for i in range(1, n):
        x = a + i * h
        seg = next(j for j in range(len(x_nodes)-1)
                   if x >= x_nodes[j] and x < x_nodes[j+1])
        coeff_x = coeff_lin(y, x_nodes)[seg]
        if i % 2 == 0:
            integral += 2 * func(x, *coeff_x)
        else:
            integral += 4 * func(x, *coeff_x)
    integral *= h / 3
    return integral


def functional_value(y_nodes: np.ndarray, x_nodes: np.ndarray,
                     A: float, B: float, n_integr: int) -> float:
    s = 0
    y = np.concatenate(([A], y_nodes, [B]))
    coeffs = coeff_lin(y, x_nodes)
    for i in range(len(x_nodes) - 1):
        s += composite_simpson(coeffs[i], func, x_nodes[i],
                               x_nodes[i + 1], n_integr, x_nodes, y)
    return s


def get_optim_nodes_efficient(f: callable, y_nodes: np.ndarray,
                              A: float, B: float, x_nodes: np.ndarray, n_integr: int) -> Tuple[np.ndarray, float]:
    bounds = Bounds(0.0001, np.inf)
    result = minimize(f, y_nodes, args=(x_nodes, A, B, n_integr),
                      method='L-BFGS-B', bounds=bounds, options={'maxiter': 15000})
    return result.x, result.fun


def piecewise_linear(x, x_nodes, y_nodes):
    y = np.zeros_like(x)
    for i in range(len(x_nodes) - 1):
        x1, x2 = x_nodes[i], x_nodes[i + 1]
        y1, y2 = y_nodes[i], y_nodes[i + 1]
        mask = (x >= x1) & (x <= x2)
        y[mask] = y1 + (y2 - y1) * (x[mask] - x1) / (x2 - x1)
    return y


def optimize_nodes(N, A_initial, B_initial, n_integr_reduced):
    x_nodes_new = np.linspace(0, 2, N)
    initial_y_nodes_new = np.linspace(0, 1, N)[1:-1]

    optim_result_new_efficient = get_optim_nodes_efficient(
        functional_value,
        initial_y_nodes_new,
        A_initial,
        B_initial,
        x_nodes_new,
        n_integr_reduced
    )

    optimized_y_nodes_new_efficient = np.insert(optim_result_new_efficient[0], [
                                                0, len(optim_result_new_efficient[0])], [A_initial, B_initial])
    return x_nodes_new, optimized_y_nodes_new_efficient


def plot_graph(x_nodes_new, optimized_y_nodes_new_efficient, t_values, C, T):
    x_cycloid, y_cycloid = cycloid(t_values, C)
    x_interp = np.linspace(0, 2, 100)
    y_interp_new_efficient = piecewise_linear(
        x_interp, x_nodes_new, optimized_y_nodes_new_efficient)

    plt.figure(figsize=(8, 5))

    plt.plot(x_cycloid, y_cycloid, '--', color='#1E90FF',
             label='Analytical solution (cycloid)')

    plt.plot(x_interp, y_interp_new_efficient, '#DC143C',
             label='Optimized')

    plt.plot([0, 2], [0, 1], 'grey', label='First guess')
    plt.scatter([0, 2], [0, 1], color='black', zorder=3,
                label=r'$A(a, y_a) = (0, 0),~B(b, y_b) = (2, 1)$')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def lint(x, x_nodes, coeffs):
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x <= x_nodes[i + 1]:
            return coeffs[i][0] * x + coeffs[i][1]
    return None


def error_optim(y_nodes):
    # Задаем константы, необходимые для построения циклоиды
    t = np.linspace(0, T, 10**2)

    # Интерполируем циклоиду
    x_cnodes = [C * (t_ - np.sin(2 * t_) / 2) for t_ in t]
    y_cnodes = [C * (1 / 2 - np.cos(2 * t_) / 2) for t_ in t]
    cmatrix = coeff_lin(y_cnodes, x_cnodes)
    analytical = [lint(x, x_cnodes, cmatrix) for x in x_cnodes]

    # Интерполируем оптимизированную функцию
    # Добавляем 2 для граничных условий
    x_nodes = np.linspace(0, 2, len(y_nodes) + 2)
    # Добавляем граничные условия
    y_nodes_with_boundaries = [0] + list(y_nodes) + [1]
    omatrix = coeff_lin(y_nodes_with_boundaries, x_nodes)
    optim = [lint(x, x_nodes, omatrix) for x in x_cnodes]

    # Вычисление евклидова расстояния
    return distance.euclidean(optim, analytical)


def compute_errors(N_range, n_range, A_initial, B_initial):
    # Инициализируем массивы для шагов интерполяции и интегрирования
    h_interp_array = 2 / (N_range - 1)
    h_integr_array = 2 / (n_range - 1)

    # Инициализируем матрицу для хранения ошибок
    error_matrix = np.zeros((len(N_range), len(n_range)))

    # Цикл по всем значениям N и n для вычисления ошибок
    minutes = 0
    for i, N in enumerate(N_range):
        start_time = time.time()
        for j, n in enumerate(n_range):
            print(N, n)
            # Создаем узлы x и начальные значения y
            x_nodes = np.linspace(0, 2, N)
            initial_y_nodes = np.linspace(
                0, 1, N)[1:-1]  # исключаем крайние точки

            # Оптимизация узлов y
            y_optimized, _ = get_optim_nodes_efficient(
                functional_value, initial_y_nodes, A_initial, B_initial, x_nodes, n)

            # Вычисление ошибки
            error_matrix[i, j] = error_optim(y_optimized)
        end_time = time.time()
        minutes += (end_time - start_time) / 60
        print(f"Время выполнения для N={N}: {minutes} минут")

    print(error_matrix)
    return h_interp_array, h_integr_array, error_matrix


def plot_3d_graph(h_interp_array, h_integr_array, error_matrix, elev=20):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Создаем сетку координат для графика
    h_integr_grid, h_interp_grid = np.meshgrid(h_integr_array, h_interp_array)

    # Построение поверхности
    surf = ax.plot_surface(h_interp_grid, h_integr_grid,
                           error_matrix, cmap=cm.Reds)

    # Подписи осей
    ax.set_xlabel('Шаг интерполяции')
    ax.invert_xaxis()
    ax.set_ylabel('Шаг интегрирования')
    ax.set_zlabel('Погрешность')

    ax.view_init(elev=elev, azim=-45)

    # Показываем график
    plt.tight_layout()
    plt.show()


def plot_contour(h_interp_array, h_integr_array, error_matrix):
    fig, ax = plt.subplots(figsize=(7, 7))

    # Создаем сетку координат для графика
    h_integr_grid, h_interp_grid = np.meshgrid(h_integr_array, h_interp_array)

    # Построение линий уровня
    contour = ax.contour(h_interp_grid, h_integr_grid,
                         error_matrix, cmap=cm.Reds)

    # Добавляем значения на линии контура
    ax.clabel(contour, inline=True, fontsize=8)

    # Подписи осей
    ax.set_xlabel('Шаг интерполяции')
    ax.set_ylabel('Шаг интегрирования')

    # Показываем график
    plt.tight_layout()
    plt.show()


def plot_constatns_graph_const_N(N, n):
    N_range = np.arange(N, N+1)
    n_range = np.arange(3, n+1)

    A_initial, B_initial = 0, 1

    h_interp_array, h_integr_array, error_matrix = compute_errors(
        N_range, n_range, A_initial, B_initial)

    plt.figure(figsize=(7, 7))
    plt.scatter(h_integr_array,
                error_matrix[0], s=10, label=r'$E(H=const, h)$', color='#1E90FF')
    plt.plot(h_integr_array, [sys.float_info.epsilon for _ in h_integr_array],
             lw=2, color='orange', zorder=0, label=r'$\epsilon$')
    plt.legend()
    plt.grid()
    plt.loglog()
    plt.show()


def plot_constatns_graph_const_n(N, n):
    N_range = np.arange(3, N+1)
    n_range = np.arange(n, n+1)

    A_initial, B_initial = 0, 1

    h_interp_array, h_integr_array, error_matrix = compute_errors(
        N_range, n_range, A_initial, B_initial)

    plt.figure(figsize=(7, 7))
    plt.scatter(h_interp_array, error_matrix.flatten(), s=10,
                label=r'$E(H, h=const)$', color='#1E90FF')
    plt.plot(h_interp_array, [sys.float_info.epsilon for _ in h_interp_array],
             lw=2, color='orange', zorder=0, label=r'$\epsilon$')
    plt.legend()
    plt.grid()
    plt.loglog()
    plt.show()
