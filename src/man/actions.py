import numpy as np
from scipy.optimize import least_squares


def find_affine_transformation(src_points, dst_points):
    """
    Find the affine transformation matrix that maps src_points to dst_points.

    :param src_points: Nx2 array of source points
    :param dst_points: Nx2 array of destination points
    :return: 3x3 affine transformation matrix
    """

    def affine_error(p, src, dst):
        A = np.array([[p[0], p[1], p[2]],
                      [p[3], p[4], p[5]],
                      [0, 0, 1]])
        src_homogeneous = np.hstack((src, np.ones((src.shape[0], 1))))
        transformed = np.dot(src_homogeneous, A.T)
        return (transformed[:, :2] - dst).ravel()

    # Initial guess for the affine transformation parameters
    initial_guess = [1, 0, 0, 0, 1, 0]

    # Perform least squares optimization
    result = least_squares(affine_error, initial_guess, args=(src_points, dst_points))



    # Construct the affine transformation matrix
    A = np.array([[result.x[0], result.x[1], result.x[2]],
                  [result.x[3], result.x[4], result.x[5]],
                  [0, 0, 1]])

    #print(affine_error(result, src_points, dst_points))

    return A, result.cost



def decompose_affine_matrix(A):
    """
    Decompose the affine transformation matrix into its components, including mirroring.

    :param A: 3x3 affine transformation matrix
    :return: Dictionary containing individual transformation components
    """
    # Extract translation
    tx, ty = A[0, 2], A[1, 2]

    # Check for mirroring
    det = np.linalg.det(A[:2, :2])
    mirrored = det < 0

    # If mirrored, flip the y-axis to proceed with normal decomposition
    if mirrored:
        A = A.copy()
        A[1, :2] = -A[1, :2]

    # Extract scale and shear
    sx = np.sqrt(A[0, 0] ** 2 + A[1, 0] ** 2)
    sy = np.sqrt(A[0, 1] ** 2 + A[1, 1] ** 2)
    shear_x = np.arctan2(A[0, 1], A[0, 0])
    shear_y = np.arctan2(-A[1, 0], A[1, 1])

    # Extract rotation
    rotation = np.arctan2(A[1, 0], A[0, 0])

    return {
        "translation": (tx, ty),
        "scale": (sx, sy),
        "shear": (shear_x, shear_y),
        "rotation": rotation,
        "mirrored": mirrored
    }


def find_rigid_transformation(src_points, dst_points):
    """
    Find the rigid transformation (rotation, mirroring, translation) that maps src_points to dst_points.

    :param src_points: Nx2 array of source points
    :param dst_points: Nx2 array of destination points
    :return: 3x3 transformation matrix
    """

    def rigid_error(p, src, dst):
        theta, tx, ty = p
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        T = np.array([tx, ty])
        transformed = np.dot(src, R.T) + T
        return (transformed - dst).ravel()

    # Initial guess for the transformation parameters
    initial_guess = [0, 0, 0]  # rotation angle, tx, ty

    # Perform least squares optimization
    result = least_squares(rigid_error, initial_guess, args=(src_points, dst_points))

    # Construct the transformation matrix
    theta, tx, ty = result.x
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, tx],
                  [s, c, ty],
                  [0, 0, 1]])

    return R, result.cost


def decompose_rigid_matrix(R):
    """
    Decompose the rigid transformation matrix into its components.

    :param R: 3x3 rigid transformation matrix
    :return: Dictionary containing individual transformation components
    """
    # Extract translation
    tx, ty = R[0, 2], R[1, 2]

    # Check for mirroring
    det = np.linalg.det(R[:2, :2])
    mirrored = det < 0

    # Extract rotation
    if mirrored:
        rotation = np.arctan2(-R[1, 0], -R[0, 0])
    else:
        rotation = np.arctan2(R[1, 0], R[0, 0])

    return {
        "translation": (tx, ty),
        "rotation": rotation,
        "mirrored": mirrored
    }


def find_discrete_rigid_transformation(src_points, dst_points):
    """
    Find the discrete rigid transformation (90-degree rotation, mirroring, translation)
    that best maps src_points to dst_points.

    :param src_points: Nx2 array of source points
    :param dst_points: Nx2 array of destination points
    :return: tuple (R, t, error) where R is 2x2 rotation/reflection matrix,
             t is 2x1 translation vector, and error is the transformation error
    """
    # Center the points
    src_center = np.mean(src_points, axis=0)
    dst_center = np.mean(dst_points, axis=0)
    src_centered = src_points - src_center
    dst_centered = dst_points - dst_center

    # Possible rotation matrices (0, 90, 180, 270 degrees)
    rotations = [
        np.array([[1, 0], [0, 1]]),  # 0 degrees
        np.array([[0, -1], [1, 0]]),  # 90 degrees
        np.array([[-1, 0], [0, -1]]),  # 180 degrees
        np.array([[0, 1], [-1, 0]])  # 270 degrees
    ]

    # Possible reflection matrices
    reflections = [
        np.array([[1, 0], [0, 1]]),  # No reflection
        np.array([[-1, 0], [0, 1]]),  # Reflection across y-axis
        np.array([[1, 0], [0, -1]]),  # Reflection across x-axis
        np.array([[-1, 0], [0, -1]])  # Reflection across both axes
    ]

    best_R = None
    best_error = float('inf')

    for rot in rotations:
        for ref in reflections:
            R = np.dot(rot, ref)
            transformed = np.dot(src_centered, R.T)
            error = np.sum((transformed - dst_centered) ** 2)

            if error < best_error:
                best_error = error
                best_R = R

    # Calculate translation
    t = dst_center - np.dot(src_center, best_R.T)

    return best_R, t, best_error


def decompose_discrete_rigid_matrix(R, t):
    """
    Decompose the discrete rigid transformation into its components.

    :param R: 2x2 rotation/reflection matrix
    :param t: 2x1 translation vector
    :return: Dictionary containing individual transformation components
    """
    # Determine rotation
    if np.allclose(R, np.eye(2)) or np.allclose(R, -np.eye(2)):
        rotation = 0
    elif np.allclose(R, np.array([[0, -1], [1, 0]])) or np.allclose(R, np.array([[0, 1], [-1, 0]])):
        rotation = 90
    elif np.allclose(abs(R), np.eye(2)):
        rotation = 180
    else:
        rotation = 270

    # Determine mirroring
    mirrored_x = R[0, 0] < 0 if rotation in [0, 180] else R[1, 0] < 0
    mirrored_y = R[1, 1] < 0 if rotation in [0, 180] else R[0, 1] > 0

    return {
        "translation": tuple(t),
        "rotation": rotation,
        "mirrored_x": mirrored_x,
        "mirrored_y": mirrored_y
    }

# # Example usage
# src_points = np.array([[0, 0], [1, 0], [0, 1]])
# dst_points = np.array([[1, 1], [2, 1], [1, 3]])
#
# A = find_affine_transformation(src_points, dst_points)
# components = decompose_affine_matrix(A)
#
# print("Affine Transformation Matrix:")
# print(A)
# print("\nDecomposed Components:")
# for key, value in components.items():
#     print(f"{key}: {value}")