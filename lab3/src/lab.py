import numpy as np
import matplotlib.pyplot as plt
import random

def f(t, y):
	return np.array([y[1], -0.1*y[1] - np.sin(y[0]) + np.cos(t)])


def runge_kutta(x_0, t_n, f, h):
	steps = int(t_n / h)
	t = np.linspace(0, t_n, steps+1)
	w_r = np.zeros([steps+1, len(x_0)])
	w_r[0] = x_0

	for i in range(steps):
		k_1 = h*f(t[i], w_r[i])
		k_2 = h*f(t[i] + h/2, w_r[i] + k_1/2)
		k_3 = h*f(t[i] + h/2, w_r[i] + k_2/2)
		k_4 = h*f(t[i] + h, w_r[i] + k_3)
		w_r[i+1] = w_r[i] + (k_1 + 2*k_2 + 2*k_3 + k_4)/6

	return t, w_r

def milne_simpson(x_0, t_n, f, h):
	steps = int(t_n / h)
	t = np.linspace(0, t_n, steps+1)
	w_m = np.zeros([steps+1, len(x_0)])
	w_m[0] = x_0
	t_start, w_start = runge_kutta(x_0, t_n, f, h)
	w_m[1] = w_start[1]
	w_m[2] = w_start[2]
	w_m[3] = w_start[3]

	for i in range(3, steps):
		w_cor = w_m[i-3] + 4 * h *(2*f(t[i] ,w_m[i]) - f(t[i-1], w_m[i-1]) + 2*f(t[i-2], w_m[i-2])) / 3
		w_m[i+1] = w_m[i-1] + h * (f(t[i+1], w_cor) + 4*f(t[i], w_m[i]) + f(t[i-1], w_m[i-1])) / 3

	return t, w_m

# Тестирование методов с разными начальными условиями
h = 0.1
t_n = 150  # Пример конечного времени
theta_0 = 0  # Начальное значение theta

cond = []
for _ in range(15):
	dtheta_0 = random.uniform(1.85, 2.1)  # Случайное начальное условие для dtheta/dt
	x_0 = np.array([theta_0, dtheta_0])
	cond.append(x_0)

plt.figure(figsize=(10, 5))
for idx, x_0 in enumerate(cond):
	t_rk, w_rk = runge_kutta(x_0, t_n, f, h)
	label = f'При dθ₀/dt={x_0[1]:.2f}'  # Форматирование строки с округлением dtheta_0 до двух знаков после запятой
	plt.plot(t_rk, w_rk[:, 0], label=label)
plt.title("Метод Рунге-Кутты")
plt.xlabel("t", fontsize = 16)
plt.ylabel("θ(t)", fontsize = 16)
plt.legend(loc='center left', bbox_to_anchor=(1.00, 0.4))
plt.tight_layout()
plt.show()

# График для метода Милна-Симпсона
plt.figure(figsize=(10, 5))
for idx, x_0 in enumerate(cond):
	t_ms, w_ms = milne_simpson(x_0, t_n, f, h)
	label = f'При dθ₀/dt={x_0[1]:.2f}'  # Аналогично для Милна-Симпсона
	plt.plot(t_ms, w_ms[:, 0], label=label)
plt.title("Метод Милна-Симпсона")
plt.xlabel("t", fontsize = 16)
plt.ylabel("θ(t)", fontsize = 16)
plt.legend(loc='center left', bbox_to_anchor=(1.00, 0.4))
plt.tight_layout()

plt.show()

# Фиксированное начальное условие
fixed_dtheta_0 = 1.85
fixed_x_0 = np.array([theta_0, fixed_dtheta_0])

# Список различных значений шага h
h_values_rk = [1.05, 0.98, 0.91, 0.84, 0.77, 0.7, 0.63, 0.56, 0.49, 0.42, 0.35, 0.28, 0.21, 0.14, 0.1]

# Графики для метода Рунге-Кутты с разными h
plt.figure(figsize=(12, 6))
for h in h_values_rk:
    t_rk, w_rk = runge_kutta(fixed_x_0, t_n, f, h)
    plt.plot(t_rk, w_rk[:, 0], label=f'RK4, h={h}')
plt.title("Метод Рунге-Кутты с разными h")
plt.xlabel("Время t")
plt.ylabel("θ(t)")
plt.legend()
plt.show()

h_values_ms = [0.49, 0.42, 0.35, 0.28, 0.21, 0.14, 0.1]
# Графики для метода Милна-Симпсона с разными h
plt.figure(figsize=(12, 6))
for h in h_values_ms:
    t_ms, w_ms = milne_simpson(fixed_x_0, t_n, f, h)
    plt.plot(t_ms, w_ms[:, 0], label=f'Milne-Simpson, h={h}')
plt.title("Метод Милна-Симпсона с разными h")
plt.xlabel("Время t")
plt.ylabel("θ(t)")
plt.legend()
plt.show()
