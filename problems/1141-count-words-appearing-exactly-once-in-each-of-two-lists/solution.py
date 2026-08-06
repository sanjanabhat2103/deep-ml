def count_common_unique(list1, list2):
    # list1: list of strings
    # list2: list of strings
    # return an integer
    count1 = {}
    count2 = {}
    for item in list1:
        count1[item] = count1.get(item, 0) + 1
    for item in list2:
        count2[item] = count2.get(item, 0) + 1
    common_unique = 0
    for item in count1:
        if item in count2 and count1[item] == 1 and count2[item] == 1:
            common_unique += 1
    return common_unique