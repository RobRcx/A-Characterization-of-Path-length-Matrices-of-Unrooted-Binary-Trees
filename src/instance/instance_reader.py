import numpy as np


def generate(n=5, dmax=1):
    D = np.random.random((n, n))
    assert D.shape[0] == D.shape[1]
    assert D.shape[0] == n
    for i in range(n):
        D[i][i] = 0
    for i in range(n):
        for j in range(n):
            D[i][j] = dmax * abs(D[i][j])
    for i in range(n):
        for j in range(i + 1, n):
            D[i][j] = D[j][i]
    return D


def test():
    n = 10
    D = generate(n=n)
    for i in range(n):
        for j in range(n):
            print(str(D[i][j]) + ' ', end='')
        print()


def _read_matrix(lines):
    n = int(lines[0])
    d = np.zeros((n, n))
    i = 0
    for l in lines[1:]:
        spl = l.split(' ')
        for j in range(min(n, len(spl))):
            d[i][j] = float(spl[j])
        i += 1
    return n, d

def read_instance(filename, skiprows=0):
    f = open(filename, "r")
    lines = f.readlines()
    n, d = _read_matrix(lines[skiprows:])
    f.close()
    return n, d


