def matrixmul(a:list[list[int|float]], b:list[list[int|float]])-> list[list[int|float]]:
    
    p = len(a)
    q = len(a[0])
    r = len(b)
    s = len(b[0])
    if q != r:
        return -1
    c = [[0 for j in range(s)] for i in range(p)]
    for i in range(p):
        for j in range(s):
            for k in range(q):
                c[i][j] += a[i][k] * b[k][j]
    return c