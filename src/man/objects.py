import numpy as np

from collections import deque
import matplotlib.pyplot as plt




def plot_objects(image, objects):
    plt.figure(figsize=(12, 5))

    # Plot original image
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Plot labeled objects
    plt.subplot(122)
    labeled_image = np.zeros_like(image)
    for i, obj in enumerate(objects, 1):
        for x, y in obj['pixels']:
            labeled_image[x, y] = i

    plt.imshow(labeled_image, cmap='tab20')  # Using a colormap with distinct colors
    plt.title('Labeled Objects')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def find_objects(image, name, background_colors, minimum_size = 2):
    height, width = image.shape
    visited = np.zeros((height, width), dtype=bool)
    objects = []

    def is_valid(x, y):
        return 0 <= x < height and 0 <= y < width

    def bfs(start_x, start_y, color):
        queue = deque([(start_x, start_y)])
        visited[start_x][start_y] = True
        object_pixels = [(start_x, start_y)]

        # 8-connectivity: include diagonal directions
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny) and not visited[nx][ny] and image[nx][ny] == color:
                    queue.append((nx, ny))
                    visited[nx][ny] = True
                    object_pixels.append((nx, ny))

        return object_pixels

    for i in range(height):
        for j in range(width):
            if not visited[i][j] and image[i][j] not in background_colors:
                object_pixels = bfs(i, j, image[i][j])

                if(len(object_pixels) >= minimum_size):
                    objects.append({
                        'name' : name,
                        'colour': image[i][j],
                        'pixels': object_pixels,
                        'size': len(object_pixels)
                    })

    return objects
