import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def matrix_generation(n):
    matrix = np.random.uniform(-1.0001, 1, (n, n)).astype(np.float32)
    while np.linalg.det(matrix) == 0:
        matrix = np.random.uniform(-1.0001, 1, (n, n)).astype(np.float32)
    return matrix


def three_diag(n):
    matrix = np.zeros((n, n), dtype=np.float32)
    print(matrix[0][0].dtype)
    matrix[0][0] = np.random.uniform(-1.0001, 1)
    matrix[0][1] = np.random.uniform(-1.0001, 1)
    print(matrix[0][0].dtype)
    for i in range(1, n - 1):
        for j in range(i - 1, i + 2):
            matrix[i][j] = np.random.uniform(-1.0001, 1)
    matrix[n - 1][n - 2] = np.random.uniform(-1.0001, 1)
    matrix[n - 1][n - 1] = np.random.uniform(-1.0001, 1)
    return matrix


def gauss(A, b, pivoting):
    n = len(A)
    if pivoting:
        for i in range(n - 1):
            column_max = i + np.argmax(np.abs(A[i:, i]), axis=0)
            A_temp = np.array([A[i], A[column_max]])
            A[i] = A_temp[1]
            A[column_max] = A_temp[0]

    for row in range(n):
        for i in range(row + 1, n):
            frac = A[i][row] / A[row][row]
            for j in range(n):
                A[i][j] = A[i][j] - frac * A[row][j]
            b[i] = b[i] - frac * b[row]

    x = np.empty(n)
    for i in range(n - 1, -1, -1):
        summ = 0
        for j in range(i + 1, n):
            summ += A[i][j] * x[j]
        x[i] = (b[i] - summ) / A[i][i]
    return x


def thomas(A, b):
    n = len(A)
    gamma = np.empty(n, dtype=np.float32)
    beta = np.empty(n, dtype=np.float32)
    x = np.empty(n, dtype=np.float32)
    gamma[0] = beta[0] = 0
    for i in range(n - 1):
        gamma[i + 1] = -A[i][i + 1] / (A[i][i - 1] * gamma[i] + A[i][i])
        beta[i + 1] = (b[i] - A[i][i - 1] * beta[i]) / (A[i][i - 1] * gamma[i] + A[i][i])
    x[n - 1] = (b[n - 1] - A[n - 1][n - 2] * beta[n - 1]) / (A[n - 1][n - 1] + A[n - 1][n - 2] * gamma[n - 1])
    for i in range(n - 1, 0, -1):
        x[i - 1] = gamma[i] * x[i] + beta[i]
    print(A)
    print(x)
    return x


def gauss_error():
    n = 6
    count = 1000
    square_gauss = np.empty(count, dtype=np.float32)
    supremum_gauss = np.empty(count, dtype=np.float32)
    radius_gauss = np.empty(count, dtype=np.float32)
    conditional_gauss = np.empty(count, dtype=np.float32)
    for i in range(count):
        A = matrix_generation(n)
        A_copy1 = np.array(A)
        A_copy2 = np.array(A)
        x_gauss_element = gauss(A_copy1, [1, 1, 1, 1, 1, 1], True)
        x_gauss = gauss(A_copy2, [1, 1, 1, 1, 1, 1], False)

        square_gauss[i] = np.linalg.norm(x_gauss_element - x_gauss) / np.linalg.norm(x_gauss_element)
        supremum_gauss[i] = np.linalg.norm((x_gauss_element - x_gauss), ord=np.inf) / np.linalg.norm(x_gauss_element,
                                                                                                     ord=np.inf)
        radius_gauss[i] = np.max(np.real(np.abs(np.linalg.eigvals(A))))
        conditional_gauss[i] = np.linalg.cond(A)
    plt.hist(square_gauss, np.linspace(0, 0.00001, 100), color='hotpink', edgecolor='black')
    plt.xlabel('δ')
    plt.title('Относительная погрешность для квадратичной нормы (метод Гаусса)')
    plt.show()

    plt.hist(supremum_gauss, np.linspace(0, 0.00001, 100), color='hotpink', edgecolor='black')
    plt.xlabel('δ')
    plt.title('Относительная погрешность для супремум-нормы (метод Гаусса)')
    plt.show()

    plt.hist(radius_gauss, np.linspace(0, 3, 100), color='hotpink', edgecolor='black')
    plt.title('Спектральные радиусы невырожденных матриц')
    plt.show()

    plt.hist(conditional_gauss, np.linspace(0, 1000, 100), color='hotpink', edgecolor='black')
    plt.title('Числа обусловленности невырожденных матриц')
    plt.show()


def thomas_error():
    n = 6
    count = 1000
    square_thomas = np.empty(count, dtype=np.float32)
    supremum_thomas = np.empty(count, dtype=np.float32)
    radius_thomas = np.empty(count, dtype=np.float32)
    conditional_thomas = np.empty(count, dtype=np.float32)
    for i in range(count):
        A = three_diag(n)
        A_copy = np.array(A)
        b = [1, 1, 1, 1, 1, 1]
        x_thomas = thomas(A, b)
        x_gauss_element = gauss(A_copy, b, True)
        print("gauss", x_gauss_element)
        square_thomas[i] = np.linalg.norm(x_gauss_element - x_thomas) / np.linalg.norm(x_gauss_element)
        supremum_thomas[i] = np.linalg.norm((x_gauss_element - x_thomas), ord=np.inf) / np.linalg.norm(x_gauss_element,
                                                                                                      ord=np.inf)
        radius_thomas[i] = np.max(np.real(np.abs(np.linalg.eigvals(A))))
        conditional_thomas[i] = np.linalg.cond(A)
    plt.hist(square_thomas, np.linspace(0, 0.00001, 100), color='hotpink', edgecolor='black')
    plt.xlabel('δ')
    plt.title('Относительная погрешность для квадратичной нормы (метод прогонки)')
    plt.show()

    plt.hist(supremum_thomas, np.linspace(0, 0.00001, 100), color='hotpink', edgecolor='black')
    plt.xlabel('δ')
    plt.title('Относительная погрешность для супремум-нормы (метод прогонки)')
    plt.show()

    plt.hist(radius_thomas, np.linspace(0, 3, 100), color='hotpink', edgecolor='black')
    plt.title('Спектральные радиусы трехдиагональных матриц')
    plt.show()

    plt.hist(conditional_thomas, np.linspace(0, 200, 100), color='hotpink', edgecolor='black')
    plt.title('Числа обусловленности трехдиагональных матриц')
    plt.show()


def positive_generation(n):
    L = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i):
            L[i][j] = np.random.uniform(-1.0001, 1)
        L[i][i] = np.random.rand()  # положительные элементы на диагонали
    LT = np.transpose(L)
    print(LT[0][0].dtype)
    A = np.dot(L, LT)
    print(L)
    print(LT)
    print(A)
    print(np.all(np.linalg.eigvals(A) > 0))  # проверка на положительно определенность
    return A


def cholesky(A, b):
    L = np.zeros((n, n), dtype=np.float32)
    for j in range(n):
        L[j][j] = np.sqrt(A[j][j] - np.sum(L[j][:j] ** 2))
        print(np.sum(L[j][:j] ** 2))
        for i in range(j + 1, n):
            L[i][j] = 1 / L[j][j] * (A[i][j] - np.sum(L[i][:j] * L[j][:j]))
    print(L)
    LT = np.transpose(L)
    solution = np.zeros(n, dtype=np.float32)
    solution1 = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = np.float32(0.0)
        for j in range(i):
            s += L[i][j] * solution[j]
        solution[i] = (b[i] - s) / L[i][i]
    for i in range(n - 1, -1, -1):
        s = np.float32(0.0)
        for j in range(n - 1, i, -1):
            s += LT[i][j] * solution1[j]
        solution1[i] = (solution[i] - s) / LT[i][i]
    return solution1


def cholesky_error():
    n = 6
    count = 1000
    square_cholesky = np.empty(count, dtype=np.float32)
    supremum_cholesky = np.empty(count, dtype=np.float32)
    radius_cholesky = np.empty(count, dtype=np.float32)
    conditional_cholesky = np.empty(count, dtype=np.float32)
    for i in range(count):
        A = positive_generation(n)
        A_copy = np.array(A)
        x_cholesky = cholesky(A, [1, 1, 1, 1, 1, 1])
        x_gauss_element = gauss(A_copy, [1, 1, 1, 1, 1, 1], True)
        print(x_cholesky, x_gauss_element)
        square_cholesky[i] = np.linalg.norm(x_gauss_element - x_cholesky) / np.linalg.norm(x_gauss_element)
        supremum_cholesky[i] = np.linalg.norm((x_gauss_element - x_cholesky), ord=np.inf) / np.linalg.norm(
            x_gauss_element,
            ord=np.inf)
        radius_cholesky[i] = np.max(np.real(np.abs(np.linalg.eigvals(A))))
        conditional_cholesky[i] = np.linalg.cond(A)
    plt.hist(square_cholesky, np.linspace(0, 0.0001, 100), color='hotpink', edgecolor='black')
    plt.xlabel('δ')
    plt.title('Относительная погрешность для квадратичной нормы (метод Холецкого)')
    plt.show()

    plt.hist(supremum_cholesky, np.linspace(0, 0.0001, 100), color='hotpink', edgecolor='black')
    plt.xlabel('δ')
    plt.title('Относительная погрешность для супремум-нормы (метод Холецкого)')
    plt.show()

    plt.hist(radius_cholesky, np.linspace(0, 10, 100), color='hotpink', edgecolor='black')
    plt.title('Спектральные радиусы положительно определённых матриц')
    plt.show()

    plt.hist(conditional_cholesky, np.linspace(0, 100000, 100), color='hotpink', edgecolor='black')
    plt.title('Числа обусловленности положительно определённых матриц')
    plt.show()


if __name__ == '__main__':
    n = 6
    b = [1, 1, 1, 1, 1, 1]
    count = 1000

    gauss_error()
    thomas_error()
    cholesky_error()
