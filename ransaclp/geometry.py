'''
Geometry related functions.
'''
import numpy as np
from typing import Tuple

def fit_plane_svd(points):
    centroid = np.mean(points, axis=0)
    shifted_points = points - centroid
    _, _, V = np.linalg.svd(shifted_points)
    normal = V[2]
    d = -np.dot(normal, centroid)
    # Compute sum of squared errors
    errors = np.abs(np.dot(points, normal) + d)
    sse = np.sum(errors ** 2)

    return normal[0], normal[1], normal[2], d, sse

def get_best_plane_from_points_from_two_segments(segment_1: np.ndarray, segment_2: np.ndarray) -> Tuple[np.ndarray, float]:
    '''
    Computes the best fitting plane to the four points of two segments.

    :param segment_1: The first segment.
    :type segment_1: np.ndarray
    :param segment_2: The second segment.
    :type segment_2: np.ndarray
    :return: The best fitting plane and the sum of squared errors.
    :rtype: Tuple[np.ndarray, float]

    :Example:

    ::

        >>> import geometry as geom
        >>> import numpy as np
        >>> segment_1 = np.array([[0, 0, 0], [1, 0, 0]])
        >>> segment_2 = np.array([[0, 1, 0], [1, 1, 0]])
        >>> geom.get_best_plane_from_points_from_two_segments(segment_1, segment_2)
        (array([ 0.,  0.,  1., -0.]), 0.0)
        >>> # another example
        >>> segment_1 = np.array([[1, 2, 3], [4, 5, 6]])
        >>> segment_2 = np.array([[7, 8, 9], [10, 11, 12]])
        >>> geom.get_best_plane_from_points_from_two_segments(segment_1, segment_2)
        (array([ 0.81649658, -0.40824829, -0.40824829,  1.22474487]),
        1.0107280348144214e-29)
        >>> # another example 
        >>> segment_1 = np.array([[0, 0, 0], [1, 0, 0]])
        >>> segment_2 = np.array([[0, 1, 0], [1, 1, 1]])
        >>> geom.get_best_plane_from_points_from_two_segments(segment_1, segment_2)
        (array([-0.45440135, -0.45440135,  0.76618459,  0.2628552 ]),
        0.15692966918274637)

    '''
    points = np.array([segment_1[0], segment_1[1], segment_2[0], segment_2[1]])
    a, b, c, d, sse = fit_plane_svd(points)
    best_plane = np.array([a, b, c, d])
    return best_plane, sse
