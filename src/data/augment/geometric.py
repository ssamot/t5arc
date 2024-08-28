import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


# ... [Previous functions remain unchanged] ...

def scale(array: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Detect, scale the image, and replace in the original array.
    If scaling up, crop to fit. If scaling down, fill with zeros.
    """
    coords = detect_image(array)
    if coords is None:
        return array
    (top, left), (bottom, right) = coords
    subimage = array[top:bottom + 1, left:right + 1]
    original_shape = subimage.shape

    # Calculate new dimensions
    new_shape = tuple(int(dim * scale_factor) for dim in original_shape)

    # Create a new array filled with zeros, same size as the original subimage
    new_subimage = np.zeros_like(subimage)

    # Scale the subimage
    scaled = ndimage.zoom(subimage, scale_factor, order=0)

    if scale_factor > 1:  # Scaling up
        # Crop the scaled image to fit the original dimensions
        new_subimage = scaled[:original_shape[0], :original_shape[1]]
    else:  # Scaling down
        # Place the scaled image in the top-left corner of the new subimage
        new_subimage[:new_shape[0], :new_shape[1]] = scaled

    return replace_subimage(array, new_subimage, coords)


def main():
    # Example usage
    n = 64
    array = np.zeros((n, n), dtype=int)
    image = np.random.randint(1, 11, size=(4, 4))
    start = n // 2 - 2  # Center the 4x4 image
    array[start:start + 4, start:start + 4] = image

    print("Original array (showing center 10x10):")
    center = n // 2
    print(array[center - 5:center + 5, center - 5:center + 5])

    print("\nScaled up by 2x (showing center 10x10):")
    scaled_up = scale(array, 2)
    print(scaled_up[center - 5:center + 5, center - 5:center + 5])

    print("\nScaled down by 0.5x (showing center 10x10):")
    scaled_down = scale(array, 0.5)
    print(scaled_down[center - 5:center + 5, center - 5:center + 5])


if __name__ == "__main__":
    main()