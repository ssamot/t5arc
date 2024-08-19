import numpy as np


def extract_and_count_subwindows(image):
    height, width = image.shape
    total_count = 0

    # Iterate over all possible sub-window sizes
    for a in range(2, height + 1):
        for b in range(2, width + 1):
            # Iterate over all possible positions of the top-left corner of the sub-window
            for i in range(height - a + 1):
                for j in range(width - b + 1):
                    # Extract the sub-window of size a x b
                    sub_window = image[i:i + a, j:j + b]
                    # Count this sub-window
                    total_count += 1

    return total_count


# Define the size of the image
image_shape = (32, 32)

# Create a dummy image of the specified shape (you can use np.zeros or any other method to create the array)
image = np.zeros(image_shape, dtype=int)

# Calculate the total number of sub-windows
total_subwindows = extract_and_count_subwindows(image)
print("Total number of sub-windows:", total_subwindows)
