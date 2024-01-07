import numpy as np
import matplotlib.pyplot as plt


def gauss(A, b, pivoting=False):
    A = np.copy(A)
    b = np.copy(b)
    n = len(b)

    for i in range(0, n - 1):
        if pivoting:
            max_i = np.argmax(np.abs(A[:, i][i:])) + i

            b[[i, max_i]] = b[[max_i, i]]
            A[[i, max_i]] = A[[max_i, i]]

        for j in range(i + 1, n):
            b[j] = b[j] - (A[j, i] / A[i, i]) * b[i]
            A[j] = A[j] - (A[j, i] / A[i, i]) * A[i]

    x = np.zeros([n])

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.sum(A[i][i + 1:] * x[i + 1:])) / A[i, i]

    return x[:n]


def thomas(A, b):
    n = len(b)

    y = np.zeros(n + 1)
    bt = np.zeros(n + 1)

    for i in range(n):
        a = ((A[i, i - 1] if i > 0 else 0),
             A[i, i],
             (A[i, i + 1] if i < n - 1 else 0))

        y[i + 1] = -a[2] / (a[1] + y[i] * a[0])
        bt[i + 1] = (b[i] - a[0] * bt[i]) / (a[0] * y[i] + a[1])

    x = np.zeros(n + 1)

    for i in range(n - 1, -1, -1):
        x[i] = y[i + 1] * x[i + 1] + bt[i + 1]

    return x[:n]


def cholesky(A, b):
    L = np.zeros(A.shape)
    n = len(b)

    for i in range(n):
        for j in range(0, i):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

        L[i, i] = np.sqrt(A[i, i] - np.sum(L[i, :i] ** 2))

    y = np.zeros(n)

    for i in range(n):
        y[i] = (b[i] - np.sum(L[i, :i] * y[:i])) / L[i, i]

    L = np.transpose(L)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.sum(L[i, i + 1:] * x[i + 1:])) / L[i, i]

    return x


def generate_simple(size, limits):
    L = np.eye(size[0], size[1]) + np.tril(np.random.rand(size[0], size[1]), -1)
    U = np.triu(np.random.rand(size[0], size[1]), 0)

    A = np.dot(L, U)
    A = A / (np.max(np.abs(A)) * 1.1) * (limits[1] - limits[0]) + limits[0]

    return np.matrix(A, dtype=np.float32)


def generate_diagonal(size, limits):
    L = np.eye(size[0], size[1]) + np.diag(np.random.rand(size[0] - 1), -1)
    U = np.diag(np.random.rand(size[0]), 0) + np.diag(np.random.rand(size[1] - 1), 1)

    A = np.dot(L, U)
    A = (A / (np.max(np.abs(A)) * 1.1))
    A[A > 0] = A[A > 0] * (limits[1] - limits[0]) + limits[0]

    return np.matrix(A, dtype=np.float32)


def generate_positive_definite(size, limits):
    L = np.eye(size[0], size[1]) + np.tril(np.random.rand(size[0], size[1]), 0)

    A = np.dot(L, np.transpose(L))
    A = A / (np.max(np.abs(A)) * 1.1)

    return np.matrix(A, dtype=np.float32)


size = [6, 6]
limits = [-1, 1]
n = 1000

gens_b = [np.array([1] * size[0], dtype=np.float32) for i in range(n)]

universal = lambda A, b: gauss(A, b, pivoting=True)
simple = lambda A, b: gauss(A, b, pivoting=False)


def variance_values(universal_func, target_func, gens_A, scaling):
    x_universal = np.array([universal_func(gens_A[i], gens_b[i]) for i in range(n)], dtype=np.float32)
    x_target = np.array([target_func(gens_A[i], gens_b[i]) for i in range(n)], dtype=np.float32)

    variances = x_universal - x_target

    qa_variance = [
        np.sqrt(np.sum(variances[i] ** 2)) /
        np.sqrt(np.sum(x_universal[i] ** 2)) for i in range(n)
    ]
    supremum_variance = [
        np.max(np.abs(variances[i])) /
        np.max(np.abs(x_universal[i])) for i in range(n)
    ]

    hist = qa_variance
    val_range = [np.min(hist) * scaling, np.max(hist) * scaling]

    plt.hist(hist, bins=25, range=val_range)
    plt.xlabel('$\delta_2$')
    plt.ylabel('n')
    plt.grid()
    plt.show()

    hist = supremum_variance
    val_range = [np.min(hist) * scaling, np.max(hist) * scaling]

    plt.hist(hist, bins=25, range=val_range)
    plt.xlabel('$\delta_{\infty}$')
    plt.ylabel('n')
    plt.grid()
    plt.show()


gens_simple = [generate_simple(size, limits) for i in range(n)]
gens_diagonal = [generate_diagonal(size, limits) for i in range(n)]
gens_cholesky = [generate_positive_definite(size, limits) for i in range(n)]

# variance_values(universal, simple, gens_simple, 1e-03)
# variance_values(universal, thomas, gens_diagonal, 1e-03)
# variance_values(universal, cholesky, gens_cholesky, 1)


def method_condition(gens_A, scaling=[1, 1]):
    rads = [np.max(np.abs(np.linalg.eig(A)[0])) for A in gens_A]

    hist = rads
    val_range = [np.min(hist) * scaling[0], np.max(hist) * scaling[0]]

    plt.hist(hist, bins=25, range=val_range)
    plt.xlabel('$\\rho(A)$')
    plt.ylabel('n')
    plt.grid()
    plt.show()

    conds = [np.linalg.cond(A) for A in gens_A]

    hist = conds
    val_range = [np.min(hist) * scaling[1], np.max(hist) * scaling[1]]

    plt.hist(hist, bins=25, range=val_range)
    plt.xlabel('$K(A)$')
    plt.ylabel('n')
    plt.grid()
    plt.show()


method_condition(gens_simple, scaling=[1, 1e-02])
method_condition(gens_diagonal, scaling=[1, 1e-03])
method_condition(gens_cholesky)


def comparasion_evals_cond(A):
    evals = np.linalg.eigvals(A)
    cond = np.linalg.cond(A)
    print(f'Число обусловленности: {cond}, Отношение макс и мин собс. чисел: {np.max(evals) / np.min(evals)}')


comparasion_evals_cond(gens_simple[0])
comparasion_evals_cond(gens_diagonal[0])
comparasion_evals_cond(gens_cholesky[0])
