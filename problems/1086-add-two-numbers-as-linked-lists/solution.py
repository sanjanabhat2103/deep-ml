def add_two_numbers(l1, l2):
    # l1 and l2 are lists of digits in reverse order
    result = []
    carry = 0
    i = j = 0
    while i < len(l1) or j < len(l2) or carry:
        total = carry
        if i < len(l1):
            total += l1[i]
            i += 1
        if j < len(l2):
            total += l2[j]
            j += 1
        result.append(total % 10)
        carry = total // 10
    return result