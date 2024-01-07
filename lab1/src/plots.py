import numpy as np
import matplotlib.pyplot as plt


file_path = 'contour.txt'


def compute_spline(a, b, c, d, t_):
    n = len(a) - 1
    f_ = np.zeros_like(t_)

    for i in range(n):
        mask = (t_ >= i) & (t_ <= i + 1)
        dx = t_[mask] - i
        f_[mask] = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3

    return f_


# Измененная функция cubic_spline_interpolation, чтобы включить последний интервал
def cubic_spline_interpolation(t_, f_):
    n = len(t_) - 1
    h_ = np.diff(t_)

    A = np.zeros((n + 1, n + 1))
    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, n):
        A[i, i - 1] = h_[i - 1]
        A[i, i] = 2 * (h_[i - 1] + h_[i])
        A[i, i + 1] = h_[i]

    B = np.zeros(n + 1)
    for i in range(1, n):
        B[i] = 3 * ((f_[i + 1] - f_[i]) / h_[i] - (f_[i] - f_[i - 1]) / h_[i - 1])

    c = np.linalg.solve(A, B)

    a = f_
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = (a[i + 1] - a[i]) / h_[i] - h_[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h_[i])

    return a, b, c, d


# Loading dots
data = np.loadtxt('contour.txt')

# Separating (x, y)
x = data[:, 0]
y = data[:, 1]
M = 10

# Generating nodes of interpolation
x_selected = x[::M]
y_selected = y[::M]

# Computing coefficient
x_interpolated, b_x, c_x, d_x = cubic_spline_interpolation(np.arange(len(x_selected)), x_selected)
y_interpolated, b_y, c_y, d_y = cubic_spline_interpolation(np.arange(len(y_selected)), y_selected)

# Draw values
t = [i for i in range(len(x))]
h = 0.1

# Create a new array for the fragmented elements
t_dense = np.arange(0, len(x_selected) - 1, h/M)
t_dense = np.append(t_dense, float(len(t) // 10))

# Интерполирование сплайна на частых значениях t
x_tilde = compute_spline(x_interpolated, b_x, c_x, d_x, t_dense)
y_tilde = compute_spline(y_interpolated, b_y, c_y, d_y, t_dense)


# Calculate distances
rho = np.array([])
for i in range(len(x) - 1):
    dist = np.sqrt((x[i] - x_tilde[i * 10]) ** 2 + (y[i] - y_tilde[i * 10]) ** 2)
    rho = np.append(rho, dist)
rho = np.append(rho, np.sqrt((x[-1] - x_tilde[-1]) ** 2 + (y[-1] - y_tilde[-1]) ** 2))

print('Среднее отклонение = ', rho.mean())
print('Стандартное отклонение = ', rho.std())


# Depicting plots

# Plot the first set of data in the first subplot
plt.plot(x_tilde, y_tilde, c='green', label='Кубический сплайн', zorder=2)
plt.scatter(x_selected, y_selected, s=5., c='blue', marker='o', label='Выбранные точки P', zorder=3)
plt.scatter(x, y, s=3., c='#FF00FF', marker='o', label='Исходный контур', zorder=1)
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.title('Визуализация кубических сплайнов для выбранных точек P')
plt.grid(True)
plt.legend()

# Plot the second set of data in the second subplot
# axs[1].plot(t, rho)
# axs[1].set_xlabel('Ось t')
# axs[1].set_ylabel('Ось D')
# axs[1].grid(True)

# Adjust layout for better visibility
plt.tight_layout()

plt.show()
