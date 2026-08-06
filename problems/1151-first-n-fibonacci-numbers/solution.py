def first_n_fibonacci(n):
    if n < 1:
        return []
    if n == 1:
        return [0]
    fib = [0, 1]
    first = 0
    second = 1
    for i in range(2, n):
        third = first + second
        fib.append(third)
        first, second = second, third
    return fib