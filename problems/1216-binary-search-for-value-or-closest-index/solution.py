def binary_search_closest(arr, target):
    if len(arr) == 0:
        return -1
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    if right < 0:
        return left
    if left >= len(arr):
        return right
    if abs(arr[left] - target) < abs(arr[right] - target):
        return left
    return right