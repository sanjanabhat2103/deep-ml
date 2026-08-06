def calculate_brightness(img):
    if not img or not isinstance(img, list):
        return -1
    row_length = len(img[0])
    if row_length == 0:
        return -1
    for row in img:
        if not isinstance(row, list) or len(row) != row_length:
            return -1
        for pixel in row:
            if not isinstance(pixel, (int, float)) or pixel < 0 or pixel > 255:
                return -1
    total_pixels = row_length * len(img)
    brightness = sum(sum(row) for row in img) / total_pixels
    return round(brightness, 2)