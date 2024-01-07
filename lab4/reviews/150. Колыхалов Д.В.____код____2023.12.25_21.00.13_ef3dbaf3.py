import numpy as np
import matplotlib.pyplot as plt
import random
import math
MAIN_MATRIX=\
    [1.,0.,0.,0.,0.,0.],\
    [0.,1.,0.,0.,0.,0.],\
    [0.,0.,1.,0.,0.,0.],\
    [0.,0.,0.,1.,0.,0.],\
    [0.,0.,0.,0.,1.,0.],\
    [0.,0.,0.,0.,0.,1.]
ONE_MATRIX=[1., 1., 1., 1., 1., 1.]
def GENERAL_FORM(A):
    Matrix = np.array(A, dtype=np.float32)
    length = len(Matrix[0])
    for i in range(length):
        a = (random.randint(0, 400)-200)/200
        if a == 0:
            a = 0.321
        Matrix[i] *= a
    for i in range(1, 6):
        Matrix[0] += Matrix[i]
    for i in range(1, 6):
        a = (random.randint(0, 400)-200)/200
        if a == 0:
            a = 0.321
        Matrix[i] += Matrix[0]
        Matrix[i] *= a
        if abs(Matrix[i][i]) > 1:
            Matrix[i] *= 0.1
    a = (random.randint(0, 400)-200)/200
    if a == 0:
        a = 0.321
    Matrix[0] *= a
    return Matrix


def DIAGONAL_FORM(A):
    Matrix = np.array(A, dtype=np.float32)
    length = len(Matrix[0])
    for i in range(length):
        a = (random.randint(0, 400)-200)/200
        if a == 0:
            a = 0.321
        Matrix[i][i] *= a
    for i in range(0, length-1):
        Matrix[i] += Matrix[i + 1]
    for i in range(length-1, 0, -1):
        Matrix[i] += Matrix[i - 1]
    for i in range(length):
        a = (random.randint(0, 400)-200)/200
        if a == 0:
            a = 0.321
        Matrix[i] *= a
    Matrix *= 0.1
    return Matrix


def POSITVE_FORM(A):
    Matrix = np.array(A, dtype=np.float32)
    length=len(Matrix[0])
    for i in range(length):
        a = (random.randint(0, 400)-200)/200
        if a == 0:
            a = 0.321
        Matrix[i] *= a
    for i in range(1, length):
        Matrix[i] += Matrix[i - 1]
    for i in range(length):
        a = (random.randint(0, 400)-200)/200
        if a == 0:
            a = 0.321
        Matrix[i] *= a
    Matrix *= 0.1
    B_t = trnsp(Matrix)
    B_COMP = np.dot(Matrix, B_t)
    return B_COMP
def choice_main_diag(Matrix, free, i):
    length=len(Matrix[0])
    max_ = abs(Matrix[i][i])
    j_ = i
    for j in range(i, length):
        if abs(Matrix[j][i]) > max_:
            max_ = abs(Matrix[j][i])
            j_ = j
    temp_row_A = 1. * Matrix[i]
    temp_row_b = 1. * free[i]
    Matrix[i] = Matrix[j_]
    free[i] = free[j_]
    Matrix[j_] = temp_row_A
    free[j_] = temp_row_b
    return Matrix,free
def gauss(A, b, pivoting):
    Matrix=np.array(A, dtype=np.float32)
    free = np.array(b)
    length = len(Matrix[0])
    if pivoting:
        for i in range(length):
            Matrix,free=choice_main_diag(Matrix, free, i)
            for j in range(i + 1, length):
                n = Matrix[j][i]
                Matrix[j][i] = Matrix[j][i] - (Matrix[i][i] * Matrix[j][i]) / Matrix[i][i]
                free[j] = free[j] - (free[i] * n) / Matrix[i][i]
                for k in range(i + 1, length):
                    Matrix[j][k] = Matrix[j][k] - (n * Matrix[i][k]) / Matrix[i][i]
    else:
        for i in range(length):
            for j in range(i + 1, length):
                n = Matrix[j][i]
                Matrix[j][i] = Matrix[j][i] - Matrix[i][i] * Matrix[j][i] / Matrix[i][i]
                free[j] = free[j] - (n * free[i]) / Matrix[i][i]
                for k in range(i + 1, length):
                    Matrix[j][k] = Matrix[j][k] - (n * Matrix[i][k]) / Matrix[i][i]
    solution=[]*length
    solution[length-1]= free[length - 1] / Matrix[length - 1][length - 1]
    for i in range(length-2,-1,-1):
        n=0
        for j in range(i+1,length):
            n= n + solution[j] * Matrix[i][j]
        solution[i]= (free[i] - n) / Matrix[i][i]
    return solution
def thomas(A, b):
    Matrix=np.array(A, dtype=np.float32)
    length = len(Matrix[0])
    free = np.array(b)
    gamma=[]*length
    nu=[]*length
    solution=[]*length
    gamma[0]= -1*Matrix[0][1] / (Matrix[0][0])
    nu[0]= free[0] / Matrix[0][0]
    for i in range(1,length-1):
        znam=(Matrix[i][i] + Matrix[i][i - 1] * gamma[i - 1])
        gamma[i]= -1*Matrix[i][i + 1] / znam
        nu[i]= (free[i] - nu[i - 1] * Matrix[i][i - 1]) / znam
    gamma[length-1]=0
    znam = (Matrix[length - 1][length - 1] + Matrix[length - 1][length - 2] * gamma[length - 2])
    nu[length-1] = (free[length - 1] - nu[length - 2] * Matrix[length - 1][length - 2]) / znam
    solution[length-1]=nu[length-1]
    for i in range(length-2,-1,-1):
        solution[i]=(gamma[i]*solution[i+1])+nu[i]
    return solution
def trnsp(Matrix):
    length = len(Matrix[0])
    Transp_Matrix=[[]*length]*length
    Transp_Matrix = np.array(Transp_Matrix, dtype=np.float32)
    for i in range(length):
        for j in range(length):
            Transp_Matrix[j][i]=Matrix[i][j]
    return Transp_Matrix
def cholesky(A, b):
    Matrix=np.array(A, dtype=np.float32)
    length = len(Matrix[0])
    b=np.array(b)
    L=[[0]*length]*length
    L=np.array(L,dtype=np.float32)
    for i in range(length):
        for j in range(i+1):
            if i==j:
                n=0
                for k in range(i):
                    n=n+pow(L[i][k],2)
                L[i][j]= math.sqrt(Matrix[i][i] - n)
            else:
                n=0
                for k in range(j):
                    n+=L[i][k]*L[j][k]
                L[i][j]= (Matrix[i][j] - n) / L[j][j]
    L_transp=trnsp(L)
    y = [] * length
    y[0]=b[0]/L[0][0]
    for i in range(1,length):
        n =0
        for j in range(i):
            n+=y[j]*L[i][j]
        y[i]=(b[i]-n)/L[i][i]
    solution = [] * length
    solution[length - 1] = y[length - 1] / L_transp[length - 1][length - 1]
    for i in range(length-2,-1,-1):
        n=0
        for j in range(i+1,length):
            n=n+solution[j]*L_transp[i][j]
        solution[i]=(y[i]-n)/L_transp[i][i]
    return solution
def GENERATOR(method_of_creation):
    matrix=[]
    for i in range(1000):
        matrix.append(method_of_creation(MAIN_MATRIX))
    return matrix
def GENERAL_COMP(method_of_creation_matrix, f):
    Matrix=GENERATOR(method_of_creation_matrix)
    unv=[]
    spc=[]
    err_un = []
    err_sp=[]
    err=[]
    length=len(Matrix)
    for i in range(length):
        unv.append(gauss(Matrix[i], ONE_MATRIX, True))
    for i in range(length):
        spc.append(gauss(Matrix[i], ONE_MATRIX, False))
    for i in range(length):
        err_un.append(f(unv[i]))
    for i in range(length):
        err_sp.append(f(spc[i]))
    for i in range(length):
        err.append(abs((err_un[i]-err_sp[i])/err_un[i]))
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.hist(err)
    ax.grid()
    plt.xlabel('погрешность')
    plt.ylabel('количество')

def DIAGONAL_COMP(method_of_creation_matrix, f):
    Matrix=GENERATOR(method_of_creation_matrix)
    unv=[]
    spc=[]
    err_un = []
    err_sp=[]
    err=[]
    length=len(Matrix)
    for i in range(length):
        unv.append(gauss(Matrix[i], ONE_MATRIX, True))
    for i in range(length):
        spc.append(thomas(Matrix[i], ONE_MATRIX))
    for i in range(length):
        err_un.append(f(unv[i]))
    for i in range(length):
        err_sp.append(f(spc[i]))
    for i in range(length):
        err.append(abs((err_un[i]-err_sp[i])/err_un[i]))
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.hist(err)
    ax.grid()
    plt.xlabel('погрешность')
    plt.ylabel('количество')

def POSITVE_COMP(method_of_creation_matrix, f):
    Matrix =GENERATOR(method_of_creation_matrix)
    unv=[]
    spc=[]
    err_un = []
    err_sp=[]
    err=[]
    length=len(Matrix)
    for i in range(length):
        unv.append(gauss(Matrix[i], ONE_MATRIX, True))
    for i in range(length):
        spc.append(cholesky(Matrix[i], ONE_MATRIX))
    for i in range(length):
        err_un.append(f(unv[i]))
    for i in range(length):
        err_sp.append(f(spc[i]))
    for i in range(length):
        err.append(abs((err_un[i]-err_sp[i])/err_un[i]))
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.hist(err)
    ax.grid()
    plt.xlabel('погрешность')
    plt.ylabel('количество')
def GENERAL_ANL(function):
    Matrix = GENERATOR(function)
    sp_rad=[]
    num_cond=[]
    length=len(Matrix)
    for i in range(length):
        sp_rad.append(abs(max(np.linalg.eigvals(Matrix[i]),key=abs)))
    for i in range(length):
        num_cond.append(np.linalg.cond(Matrix[i]))
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.hist(sp_rad)
    ax.grid()
    plt.xlabel('спектральный радиус')
    plt.ylabel('количество')
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.hist(num_cond)
    ax.grid()
    plt.xlabel('число обусловленности')
    plt.ylabel('количество')
def OTN(num):
    min_=abs(num[0])
    max_ = abs(num[0])
    length=len(num)
    for i in range(length):
        if abs(num[i])<min_:
            min_=abs(num[i])
    for i in range(length):
        if abs(num[i])>max_:
            max_=abs(num[i])
    return max_/min_

def RADIUS_TO_SOLUTION(function):
    Matrix = function()
    radius = []
    unv=[]
    spc=[]
    err_un = []
    err_sp=[]
    err=[]
    length = len(Matrix)
    for i in range(length):
        radius.append(abs(max(np.linalg.eigvals(Matrix[i]),key=abs)))
    for i in range(length):
        unv.append(gauss(Matrix[i], ONE_MATRIX, True))
    for i in range(length):
        spc.append(cholesky(Matrix[i], ONE_MATRIX))
    for i in range(length):
        err_un.append(np.linalg.norm(unv[i]))
    for i in range(length):
        err_sp.append(np.linalg.norm(spc[i]))
    for i in range(length):
        err.append(abs((err_un[i]-err_sp[i])/err_un[i]))
    plt.scatter(radius,err)
    plt.xlabel('спектральный радиус')
    plt.ylabel('погрешность')
    plt.grid()
    plt.show()
def OTN_TO_SOLUTION(function):
    Matrix = function()
    ontos = []
    unv=[]
    spc=[]
    err_un = []
    err_sp=[]
    err=[]
    length = len(Matrix)
    for i in range(length):
        ontos.append(OTN(np.linalg.eigvals(Matrix[i])))
    for i in range(length):
        unv.append(gauss(Matrix[i], ONE_MATRIX, True))
    for i in range(length):
        spc.append(cholesky(Matrix[i], ONE_MATRIX))
    for i in range(length):
        err_un.append(np.linalg.norm(unv[i]))
    for i in range(length):
        err_sp.append(np.linalg.norm(spc[i]))
    for i in range(length):
        err.append(abs((err_un[i]-err_sp[i])/err_un[i]))
    plt.scatter(ontos,err)
    plt.xlabel('отношение собственных чисел')
    plt.ylabel('погрешность')
    plt.grid()
    plt.show()

def NUMBER_TO_SOLUTION(function):
    Matrix = function()
    num = []
    unv = []
    spc = []
    err_un = []
    err_sp = []
    err = []
    length = len(Matrix)
    for i in range(length):
        num.append(np.linalg.cond(Matrix[i]))
    for i in range(length):
        unv.append(gauss(Matrix[i], ONE_MATRIX, True))
    for i in range(length):
        spc.append(cholesky(Matrix[i], ONE_MATRIX))
    for i in range(length):
        err_un.append(np.linalg.norm(unv[i]))
    for i in range(length):
        err_sp.append(np.linalg.norm(spc[i]))
    for i in range(length):
        err.append(abs((err_un[i] - err_sp[i]) / err_un[i]))
    plt.scatter(num, err)
    plt.xlabel('число обусловленности')
    plt.ylabel('погрешность')
    plt.grid()
    plt.show()
if __name__ == '__main__':
    GENERAL_ANL(GENERAL_FORM)
    plt.show()
