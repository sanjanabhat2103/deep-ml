import numpy as np

def cross_product(a, b):
    ans = []
    i = a[1] * b[2] - a[2] * b[1]
    ans.append(i)
    j = a[2] * b[0] - a[0] * b[2]
    ans.append(j)
    k = a[0] * b[1] - a[1] * b[0]
    ans.append(k)
    return ans