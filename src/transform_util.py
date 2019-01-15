import numpy as np


def transform_coord(coord, matrix):
    """
    Transforms the given coordinate by the given matrix
    :param coord: The coordinate to transform (tuple (x, y))
    :param matrix: The matrix to transform by (3x3 numpy array)
    :return: The transformed coordinate (tuple (x, y), components casted to ints)
    """
    h0 = matrix[0, 0]
    h1 = matrix[0, 1]
    h2 = matrix[0, 2]
    h3 = matrix[1, 0]
    h4 = matrix[1, 1]
    h5 = matrix[1, 2]
    h6 = matrix[2, 0]
    h7 = matrix[2, 1]
    h8 = matrix[2, 2]

    tx = (h0 * coord[0] + h1 * coord[1] + h2)
    ty = (h3 * coord[0] + h4 * coord[1] + h5)
    tz = (h6 * coord[0] + h7 * coord[1] + h8)
    return int(tx/tz), int(ty/tz)


def get_transform_dest_array(output_size):
    """
    Returns a destination array of the desired size. This is also used to define the
    order of points necessary for cv2.getPerspectiveTransform: the order can change, but
    it must remain consistent between these two arrays.
    :param output_size: The size to make the output image ((width, height) tuple)
    :return: The destination array, suitable to feed into cv2.getPerspectiveTransform
    """
    bottom_right = [output_size[0] - 1, output_size[1] - 1]
    bottom_left = [0, output_size[1] - 1]
    top_left = [0, 0]
    top_right = [output_size[0] - 1, 0]
    return np.array(
        [bottom_right, bottom_left, top_left, top_right],
        dtype="float32")


# Note: (-1, -1) = top left, (1, 1) = bottom right
def get_corner_point(points, x_factor, y_factor):
    """
    Fetches the point furthest in the given direction out of the collection of points.
    Internally, fetches the item with the largest value of p[0][0]*x_factor + p[0][1]*y_factor
    :param points: The collection to search in. This should be in the format produced by cv2.approxPolyDP,
    a numpy array of shape [n, 1, 2] (where n is the number of points)
    :param x_factor: Which side the point should be on horizontally: -1 = left, 1 = right, 0 = no preference
    :param y_factor: Which side the point should be on vertically: -1 = top, 1 = bottom, 0 = no preference
    :return:
    """
    return max(points, key=lambda p: x_factor * p[0][0] + y_factor * p[0][1])


# Sorts the src_float_array into the order bottom right, bottom left, top left, top right
# This is the order specified in get_transform_dest_array later: the order can
# be changed, it just has to be consistent in both places
def sort_source_float_array(src_float_array):
    sorted_float_array = np.zeros_like(src_float_array)
    sorted_float_array[0] = get_corner_point(src_float_array, 1, 1)
    sorted_float_array[1] = get_corner_point(src_float_array, -1, 1)
    sorted_float_array[2] = get_corner_point(src_float_array, -1, -1)
    sorted_float_array[3] = get_corner_point(src_float_array, 1, -1)
    return sorted_float_array