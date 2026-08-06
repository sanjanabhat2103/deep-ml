def reverse_linked_list(values, method="iterative"):
    # values: list of node values in head-to-tail order
    # method: 'iterative' or 'recursive'
    # return: list of node values after reversal
    if method == "iterative":
        prev = []
        for value in values:
            prev.insert(0, value)
        return prev
    elif method == "recursive":
        def reverse_helper(lst):
            if len(lst) <= 1:
                return lst
            return reverse_helper(lst[1:]) + [lst[0]]
        return reverse_helper(values)
    else:
        return []