import numpy as np


# Измененная функция cubic_spline_interpolation, чтобы включить последний интервал
def cubic_spline_interpolation(x_, y_):
    n = len(x_) - 1
    h_ = np.diff(x_)

    A = np.zeros((n + 1, n + 1))
    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, n):
        A[i, i - 1] = h_[i - 1]
        A[i, i] = 2 * (h_[i - 1] + h_[i])
        A[i, i + 1] = h_[i]

    B = np.zeros(n + 1)
    for i in range(1, n):
        B[i] = 3 * ((y_[i + 1] - y_[i]) / h_[i] - (y_[i] - y_[i - 1]) / h_[i - 1])

    c = np.linalg.solve(A, B)

    a = y_
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = (a[i + 1] - a[i]) / h_[i] - h_[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h_[i])

    return a, b, c, d


def lab1_base(filename_in: str, factor: int, filename_out: str):
    data = np.loadtxt(filename_in)
    x = data[:, 0]
    y = data[:, 1]

    x_hat = x[::factor]
    y_hat = y[::factor]
    t_hat = np.arange(len(x_hat))

    a_x, b_x, c_x, d_x = cubic_spline_interpolation(t_hat, x_hat)
    a_y, b_y, c_y, d_y = cubic_spline_interpolation(t_hat, y_hat)

    b_x = np.append(b_x, 0)
    d_x = np.append(d_x, 0)
    b_y = np.append(b_y, 0)
    d_y = np.append(d_y, 0)

    # Соединяем коэффициенты в одну матрицу
    coeffs = np.column_stack((a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_y))

    # Сохраняем коэффициенты в файл
    np.savetxt(filename_out, coeffs, delimiter='\t')

    return coeffs


coeffs = lab1_base("contour.txt", 10, "coeffs.txt")
