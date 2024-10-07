import numpy as np

from actions import decompose_rigid_matrix, \
    find_rigid_transformation, find_discrete_rigid_transformation, decompose_discrete_rigid_matrix


def colour_link(obj1, obj2):
    # Example heuristic: link if the sum of attributes is greater than a threshold
    return obj1['colour'] == obj2['colour']


def affine(obj1, obj2):
    # if(obj1["name"] == obj2["name"]):
    #     return False
    shape_1 = np.array(obj1["pixels"]).shape
    shape_2 = np.array(obj2["pixels"]).shape
    if (shape_1 != shape_2):
        # print(shape_1, shape_2)
        return False
    R, t,  error = find_discrete_rigid_transformation(np.array(obj1["pixels"]), np.array(obj2["pixels"]))
    # print(decompose_affine_matrix(matrix))
    print(error, obj1["name"], obj2["name"])
    print(decompose_discrete_rigid_matrix(R,t))
    if (error < 0.0000000001):

        # print(obj1["pixels"])
        # print(obj2["pixels"])

        return True
    else:
        return False


def link_objects(objects, heuristic_link):
    linked_objects = []
    visited = set()  # To keep track of already linked objects

    for i in range(len(objects)):
        if i in visited:
            continue  # Skip already linked objects
        current_group = [objects[i]]
        visited.add(i)

        for j in range(0, len(objects)):
            if j in visited:
                continue
            if heuristic_link(objects[i], objects[j]):
                current_group.append(objects[j])
                visited.add(j)

        linked_objects.append(current_group)

    return linked_objects
