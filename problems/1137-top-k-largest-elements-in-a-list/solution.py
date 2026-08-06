def top_three_largest(values):
    # values: list of numbers
    # return the three largest values in descending order
    values.sort(reverse = True)
    return values[0: 3]