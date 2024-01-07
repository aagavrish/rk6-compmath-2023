import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple

from AutoDiffNum import AutoDiffNum as ADN

fig, axs = plt.subplots(2, 2, figsize=(14, 7))

settings_for_plot = {
    'OriginalPoints':
        {'color': '#A9A9A9', 'marker': 'o', 's': 5,
            'label': 'Original Points', 'zorder': 0},
    'SparsePoints':
        {'color': '#8B0000', 'marker': 'o', 's': 7,
            'label': 'Sparse Points', 'zorder': 2},
    'Spline':
        {'color': '#FA8072', 'label': 'Spline', 'zorder': 1, 'lw': 2},
    'VectorG':
        {'color': '#2F4F4F', 'zorder': 3, 'width': 0.002},
    'VectorR':
        {'color': '#708090', 'zorder': 3, 'width': 0.002},
    'Distances':
        {'color': '#8B0000', 'lw': 1}
}


def setting_plot() -> None:
    for i, ax in enumerate(axs.flatten()):
        ax.grid(color='grey', lw=0.1)
        ax.legend(loc='best')
        if i == 1:
            ax.set_xlabel('t')
            ax.set_ylabel('Distance')
        else:
            ax.set_xlabel('x')
            ax.set_ylabel('y')


# функция, считывающая точки из файла
def read_points(file_name: str) -> Tuple[np.int32, np.ndarray, np.ndarray]:
    points = np.loadtxt(file_name, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    return len(points), x, y


# функция, отображающая точки на графике
def display_points(x: np.ndarray, y: np.ndarray,
                   ax: plt.Axes, settings: dict) -> None:
    ax.scatter(x, y, **settings)


# функция, отображающая вектора на графике
def display_vectors(x: np.ndarray, y: np.ndarray,
                    dx: np.ndarray, dy: np.ndarray,
                    step: np.int32, scale: np.float64,
                    settings: dict, label: str) -> None:
    for i in np.arange(0, len(x), step):
        if i == 0:
            plt.arrow(x[i], y[i], dx[i] * scale, dy[i]
                      * scale, label=label, **settings)
        else:
            plt.arrow(x[i], y[i], dx[i] * scale, dy[i] * scale, **settings)


# функция, создающая разреженное множество
def create_sparse_set(length: np.int32, M: np.int32,
                      x: np.ndarray, y: np.ndarray) -> None:
    sparse_set = np.arange(0, length, M)
    x_sparse, y_sparse = x[sparse_set], y[sparse_set]
    return sparse_set, x_sparse, y_sparse


# функция, которая находится коэффициенты кубического сплайна
def find_coefficients(sparse_set: np.ndarray, sparse_values: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = np.diff(sparse_set)

    A_matrix = np.zeros((len(sparse_set), len(sparse_set)))
    A_matrix[0][0], A_matrix[-1][-1] = 1, 1

    for index in range(1, len(sparse_set) - 1):
        A_matrix[index][index-1] = h[index-1]
        A_matrix[index][index] = 2 * (h[index] + h[index-1])
        A_matrix[index][index+1] = h[index]

    B_matrix = np.zeros(len(sparse_set))

    for index in range(1, len(sparse_set) - 1):
        B_matrix[index] = 3/h[index] * \
            (sparse_values[index+1] - sparse_values[index]) - \
            3/h[index-1] * (sparse_values[index] - sparse_values[index-1])

    c_coef = np.linalg.solve(A_matrix, B_matrix)
    d_coef = np.array([(c_coef[index+1] - c_coef[index])/(3*h[index])
                      for index in range(len(sparse_set) - 1)])
    a_coef = np.array([sparse_values[index]
                      for index in range(len(sparse_set))])
    b_coef = np.array([(1/h[index]) *
                       (a_coef[index+1] - a_coef[index]) - (h[index]/3) *
                       (c_coef[index+1] + 2*c_coef[index])
                       for index in range(len(sparse_set) - 1)])
    return a_coef[:-1], b_coef, c_coef[:-1], d_coef


# функция для расчета значения кубического сплайна в точке
def spline(a: np.float64, b: np.float64, c: np.float64, d: np.float64,
           t: Union[np.float64, ADN], t_j: Union[np.float64, ADN]) \
        -> Union[np.float64, ADN]:
    return (t - t_j) ** 3 * d + (t - t_j) ** 2 * c + (t - t_j) * b + a


# функция, которая подсчитывает расстояние
# и выводит среднее и стандартное отклонение
def calculate_distances(val: np.ndarray, val_inter: np.ndarray,
                        length: np.int32) -> None:
    distances = np.sqrt((val[0] - val_inter[0][:length]) **
                        2 + (val[1] - val_inter[1][:length]) ** 2)
    axs[0][1].plot(distances, **settings_for_plot['Distances'], label=(
        f'Среднее отклонение: {np.mean(distances):.10f}\n'
        f'Стандартное отклонение: {np.std(distances):.10f}'))
    print(f'Mean deviation: {np.mean(distances):.10f}')
    print(f'Standard deviation: {np.std(distances):.10f}')


# функция для расчета значений кубического сплайна
def calculate_spline(sparse_set: np.ndarray,
                     x_coeffs: np.ndarray[4], y_coeffs: np.ndarray[4],
                     h: np.float64) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for index in range(len(sparse_set)-1):
        for t in np.arange(sparse_set[index], sparse_set[index+1], h):
            x.append(
                spline(x_coeffs[0][index], x_coeffs[1][index],
                       x_coeffs[2][index], x_coeffs[3][index],
                       t, sparse_set[index]))
            y.append(
                spline(y_coeffs[0][index], y_coeffs[1][index],
                       y_coeffs[2][index], y_coeffs[3][index],
                       t, sparse_set[index]))

    return np.array(x), np.array(y)


# функция, реализующая базовую часть
def lab1_base(filename_in: str, factor: int, filename_out: str) \
        -> Tuple[np.ndarray, np.ndarray[4], np.ndarray[4]]:
    length, x, y = read_points(filename_in)
    sparse_set, x_sparse, y_sparse = create_sparse_set(length, factor, x, y)

    x_coeffs = find_coefficients(sparse_set, x_sparse)
    y_coeffs = find_coefficients(sparse_set, y_sparse)

    h = 0.1
    x_inter, y_inter = calculate_spline(sparse_set, x_coeffs, y_coeffs, h)

    display_points(x, y, axs[0][0], settings={
                   **settings_for_plot['OriginalPoints']})

    current_axes = [axs[0][0], axs[1][0], axs[1][1]]
    for ax in current_axes:
        display_points(x_sparse, y_sparse, ax, settings={
            **settings_for_plot['SparsePoints']})

    calculate_distances((x_inter[::10], y_inter[::10]),
                        (x, y), length//factor*factor)

    axs[1][0].plot(x_inter, y_inter, **settings_for_plot['Spline'])
    axs[1][1].plot(x_inter, y_inter, **settings_for_plot['Spline'])

    np.savetxt(filename_out, np.c_[x_coeffs[0], x_coeffs[1],
                                   x_coeffs[2], x_coeffs[3],
                                   y_coeffs[0], y_coeffs[1],
                                   y_coeffs[2], y_coeffs[3]],
               delimiter=' ', fmt='%+.10e')

    return sparse_set, x_coeffs, y_coeffs


# функция, которая считает первую производную кубического сплайна
def calculate_first_derivative(sparse_set: np.ndarray,
                               x_coeffs: np.ndarray, y_coeffs: np.ndarray, factor: np.int32) \
        -> Tuple[List[ADN], List[ADN]]:
    x_der, y_der = [], []
    for index in range(1, len(sparse_set)-1):
        for t in np.arange(sparse_set[index], sparse_set[index+1], factor):
            t_dual = ADN(t, 1)
            t_j_dual = ADN(sparse_set[index], 0)
            x_der.append(spline(x_coeffs[0][index], x_coeffs[1][index],
                                x_coeffs[2][index], x_coeffs[3][index],
                                t_dual, t_j_dual))
            y_der.append(spline(y_coeffs[0][index], y_coeffs[1][index],
                                y_coeffs[2][index], y_coeffs[3][index],
                                t_dual, t_j_dual))
    return x_der, y_der


# функция для построения нормали к вектору
def normalize(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_normal, y_normal = -y, x
    return x_normal, y_normal


if __name__ == '__main__':
    inputFilename, outputFilename, factor = 'contour.txt', 'coeffs.txt', 10
    sparse_set, x_coeffs, y_coeffs = lab1_base(
        inputFilename, factor, outputFilename)

    x_der, y_der = calculate_first_derivative(
        sparse_set, x_coeffs, y_coeffs, factor)
    x_normal, y_normal = normalize(
        np.array([x.dual for x in x_der]), np.array([y.dual for y in y_der]))

    step, scale = 10, 30
    display_vectors(np.array([x.real for x in x_der]),
                    np.array([y.real for y in y_der]),
                    np.array([x.dual for x in x_der]),
                    np.array([y.dual for y in y_der]),
                    step, scale,
                    settings_for_plot['VectorG'], label='G')
    display_vectors(np.array([x.real for x in x_der]),
                    np.array([y.real for y in y_der]),
                    x_normal, y_normal, step, scale,
                    settings_for_plot['VectorR'], label='R')

    setting_plot()
    plt.tight_layout()
    plt.show()
