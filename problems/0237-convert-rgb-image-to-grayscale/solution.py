import numpy as np

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.

    Args:
        image: RGB image as list or numpy array of shape (H, W, 3)
               with values in range [0, 255]

    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    try:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            return -1
        if np.any(image < 0) or np.any(image > 255):
            return -1
        grayscale = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2])
        return np.rint(grayscale).astype(int).tolist()
    except Exception:
        return -1