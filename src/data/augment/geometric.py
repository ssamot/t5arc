import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


def detect_image(array: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Detect the smaller image within the larger 2D array.
    Returns the top-left and bottom-right coordinates of the image.
    """
    rows, cols = np.where((array > 0) & (array <= 10))
    if len(rows) == 0 or len(cols) == 0:
        return None
    return (rows.min(), cols.min()), (rows.max(), cols.max())


def replace_subimage(original: np.ndarray, subimage: np.ndarray,
                     coords: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
    """
    Replace the subimage in the original array with the transformed subimage.
    The transformed subimage is positioned in the top-left corner of the original subimage area.
    """
    result = original.copy()
    (top, left), (bottom, right) = coords
    orig_height, orig_width = bottom - top + 1, right - left + 1
    sub_height, sub_width = subimage.shape

    # Create a new array of zeros with the original subimage size
    new_sub = np.zeros((orig_height, orig_width), dtype=original.dtype)

    # Calculate how much of the transformed subimage we can fit
    fit_height = min(orig_height, sub_height)
    fit_width = min(orig_width, sub_width)

    # Place the transformed subimage in the top-left corner
    new_sub[:fit_height, :fit_width] = subimage[:fit_height, :fit_width]

    # Replace in the original array
    result[top:bottom + 1, left:right + 1] = new_sub
    return result


def flip_vertical(array: np.ndarray) -> np.ndarray:
    """Detect, flip the image vertically, and replace in the original array."""
    coords = detect_image(array)
    if coords is None:
        return array
    (top, left), (bottom, right) = coords
    subimage = array[top:bottom + 1, left:right + 1]
    flipped = np.flipud(subimage)
    return replace_subimage(array, flipped, coords)


def flip_horizontal(array: np.ndarray) -> np.ndarray:
    """Detect, flip the image horizontally, and replace in the original array."""
    coords = detect_image(array)
    if coords is None:
        return array
    (top, left), (bottom, right) = coords
    subimage = array[top:bottom + 1, left:right + 1]
    flipped = np.fliplr(subimage)
    return replace_subimage(array, flipped, coords)


def rotate(array: np.ndarray, angle: int) -> np.ndarray:
    """Detect, rotate the image, and replace in the original array."""
    coords = detect_image(array)
    if coords is None:
        return array
    (top, left), (bottom, right) = coords
    subimage = array[top:bottom + 1, left:right + 1]
    rotated = ndimage.rotate(subimage, angle, reshape=True, order=0, mode='constant', cval=0)
    return replace_subimage(array, rotated, coords)


def add_noise(array: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
    """Detect, add noise to the image, and replace in the original array."""
    coords = detect_image(array)
    if coords is None:
        return array
    (top, left), (bottom, right) = coords
    subimage = array[top:bottom + 1, left:right + 1]
    noisy = subimage + noise_factor * np.random.randn(*subimage.shape)
    noisy = np.clip(noisy, 1, 10).astype(int)
    return replace_subimage(array, noisy, coords)


def adjust_brightness(array: np.ndarray, factor: float) -> np.ndarray:
    """Detect, adjust brightness of the image, and replace in the original array."""
    coords = detect_image(array)
    if coords is None:
        return array
    (top, left), (bottom, right) = coords
    subimage = array[top:bottom + 1, left:right + 1]
    brightened = np.clip(subimage * factor, 1, 10).astype(int)
    return replace_subimage(array, brightened, coords)


def shear(array: np.ndarray, shear_factor: float) -> np.ndarray:
    """Detect, shear the image, and replace in the original array."""
    coords = detect_image(array)
    if coords is None:
        return array
    (top, left), (bottom, right) = coords
    subimage = array[top:bottom + 1, left:right + 1]
    shear_matrix = np.array([[1, shear_factor], [0, 1]])
    sheared = ndimage.affine_transform(subimage, shear_matrix, order=0, mode='constant', cval=0)
    return replace_subimage(array, sheared, coords)


def scale(array: np.ndarray, scale_factor: float) -> np.ndarray:
    """Detect, scale the image, and replace in the original array."""
    coords = detect_image(array)
    if coords is None:
        return array
    (top, left), (bottom, right) = coords
    subimage = array[top:bottom + 1, left:right + 1]
    scaled = ndimage.zoom(subimage, scale_factor, order=0)
    return replace_subimage(array, scaled, coords)


def main():
    # Task usage
    n = 10
    array = np.zeros((n, n), dtype=int)
    image = np.random.randint(1, 11, size=(4, 2))  # Using a 4x2 image to demonstrate size changes
    array[3:7, 4:6] = image

    print("Original array:")
    print(array)

    print("\nVertically flipped:")
    print(flip_vertical(array))

    print("\nHorizontally flipped:")
    print(flip_horizontal(array))

    print("\nRotated 45 degrees:")
    print(rotate(array, 45))

    print("\nWith added noise:")
    print(add_noise(array))

    print("\nBrightness adjusted:")
    print(adjust_brightness(array, 1.5))

    print("\nSheared:")
    print(shear(array, 0.5))

    print("\nScaled by 1.5x:")
    print(scale(array, 1.5))

    print("\nScaled by 0.5x:")
    print(scale(array, 0.5))


if __name__ == "__main__":
    main()