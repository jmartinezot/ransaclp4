import numpy as np
import math
from . import sampling
from typing import Optional, List, Tuple, Dict

def get_how_many_below_threshold_between_line_and_points_and_their_indices(points: np.ndarray, line_two_points: np.ndarray, threshold: np.float32) -> Tuple[int, np.ndarray]:
    """
    Computes **how many points** are below a **threshold** distance from a **line** and returns their **count** and their **indices**. 

    This functions expects a **collection of points**, **two points defining the line**, and a **threshold** for inlier 
    determination. It returns a tuple containing the **number of inliers** and their **indices**. The type of the values of 
    the tuple are **int** and **np.ndarray**, respectively.

    :param points: The collection of points to measure the distance to the line.
    :type points: np.ndarray
    :param line_two_points: Two points defining the line.
    :type line_two_points: np.ndarray
    :param threshold: Maximum distance to the line.
    :type threshold: np.float32
    :return: Number of points below the threshold distance as their indices.
    :rtype: Tuple[int, np.ndarray]

    :Example:

    ::

        >>> import ransac
        >>> import numpy as np
        >>> line = np.array([[0, 0, 0], [1, 1, 1]])
        >>> points = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3], [2, 2, 4]])
        >>> threshold = 1
        >>> count, indices_below = ransac.get_how_many_below_threshold_between_line_and_points_and_their_indices(points, line, threshold)
        >>> count
        5
        >>> indices_below
        array([0, 1, 2, 3, 4])
        >>> points[indices_below]
        array([[-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1],
        [ 2,  2,  2],
        [ 2,  2,  3]])

    """
    B = line_two_points[0]
    C = line_two_points[1]
    # the distance between a point A and the line defined by the points B and C can be computed as 
    # magnitude(cross(A - B, C - B)) / magnitude(C - B)
    # https://math.stackexchange.com/questions/1905533/find-perpendicular-distance-from-point-to-line-in-3d

    cross_product = np.cross(points - B, C - B)
    magnitude_cross_product = np.linalg.norm(cross_product, axis=1)
    magnitude_C_minus_B = np.linalg.norm(C - B)
    distance = magnitude_cross_product / magnitude_C_minus_B

    indices_inliers = np.array([index for index, value in enumerate(distance) if value <= threshold], dtype=np.int64)
    return len(indices_inliers), indices_inliers

def get_how_many_below_threshold_between_plane_and_points_and_their_indices(points: np.ndarray, plane: np.ndarray, threshold: np.float32) -> Tuple[int, np.ndarray]:
    """
    Computes **how many points** are below a **threshold** distance from a **plane** and returns their **count** and their **indices**.

    This functions expects a **collection of points**, **four parameters defining the plane**, and a **threshold** for inlier
    determination. It returns a tuple containing the **number of inliers** and their **indices**. The type of the values of
    the tuple are **int** and **np.ndarray**, respectively.

    :param points: The collection of points to measure the distance to the line.
    :type points: np.ndarray
    :param plane: Four parameters defining the plane in the form Ax + By + Cz + D = 0.
    :type plane: np.ndarray
    :param threshold: Maximum distance to the line.
    :type threshold: np.float32
    :return: Number of points below the threshold distance as their indices.
    :rtype: Tuple[int, np.ndarray]

    :Example:

    ::

        >>> import ransac
        >>> import numpy as np
        >>> plane = np.array([1, 1, 1, 0])
        >>> points = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3], [2, 2, 4]])
        >>> threshold = 1
        >>> count, indices = ransac.get_how_many_below_threshold_between_plane_and_points_and_their_indices(points, plane, threshold)
        >>> count
        1
        >>> indices
        array([1])
        >>> points[indices]
        array([[0, 0, 0]])
    """
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]
    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]
    # the distance between the point P with coordinates (xo, yo, zo) and the given plane with equation Ax + By + Cz = D, 
    # is given by |Axo + Byo+ Czo + D|/√(A^2 + B^2 + C^2).
    denominator = math.sqrt(a * a + b * b + c * c)
    optimized_threshold = threshold * denominator  # for not computing the division for each point
    distance = np.abs(points_x * a + points_y * b + points_z * c + d)
    # point_indices = np.where(distance <= optimized_threshold)[0]
    indices_inliers = np.array([index for index, value in enumerate(distance) if value <= optimized_threshold], dtype=np.int64)
    return len(indices_inliers), indices_inliers

def get_ransac_line_iteration_results(points: np.ndarray, threshold: float, len_points:Optional[int]=None, seed: Optional[int]=None) -> dict:
    """
    Returns the results of **one iteration** of the **RANSAC** algorithm for **line fitting**.

    This functions expects a **collection of points**, the **number of points** in the collection, the **maximum distance** 
    from a point to the line for it to be considered an inlier, and the **seed** to initialize the random number generator.
    It returns a dictionary containing the **current line parameters**, **number of inliers**, and **their indices**. The keys 
    of the dictionary are **current_random_points**, **"current_line"**, **"threshold"**, **"number_inliers"**, and **"indices_inliers"**, respectively.
    
    :param points: The collection of points to fit the line to.
    :type points: np.ndarray
    :param threshold: The maximum distance from a point to the line for it to be considered an inlier.
    :type threshold: float
    :param len_points: The number of points in the collection of points.
    :type len_points: Optional[int]
    :return: A dictionary containing the current line parameters, number of inliers, and their indices.
    :rtype: dict
    """
    if len_points is None:
        len_points = len(points)
    if seed is not None:
        np.random.seed(seed)
    current_random_points = sampling.sampling_np_arrays_from_enumerable(points, cardinality_of_np_arrays=2, number_of_np_arrays=1, num_source_elems=len_points, seed=seed)[0]
    current_line = current_random_points # the easiest way to get the line parameters
    how_many_in_line, current_point_indices = get_how_many_below_threshold_between_line_and_points_and_their_indices(points, current_line, threshold)
    # print(len_points, current_random_points, current_plane, how_many_in_plane)
    return {"current_random_points": current_random_points, "current_line": current_line, "threshold": threshold, "number_inliers": how_many_in_line, "indices_inliers": current_point_indices}


def get_fitting_data_from_list_planes(points: np.ndarray, list_planes: List[np.ndarray], threshold: float) -> List[Dict]:
    '''
    Returns a list of dictionaries containing the **plane parameters**, **number of inliers**, and **their indices** for 
    **each plane in a given list**.

    It takes a collection of points, a list of plane equations, and a threshold for inlier determination. This function
    expects a **collection of points**, a **list of plane equations**, and a **threshold** for inlier determination.
    It returns a list where each entry corresponds to a plane and contains a dictionary with the plane's parameters, 
    the count of inliers, and the indices of these inliers. The keys of the dictionary are **"plane"**, **"number_inliers"**, 
    and **"indices_inliers"**, respectively. The type of the values of the dictionary are **np.ndarray**, **int**, 
    and **np.ndarray**, respectively.

    :param points: The collection of points to fit the plane to.
    :type points: np.ndarray
    :param list_planes: The list of planes to fit to the points.
    :type list_planes: List[np.ndarray]
    :param threshold: The maximum distance from a point to the plane for it to be considered an inlier.
    :type threshold: float
    :return: A list of dictionaries containing the plane parameters, number of inliers, and their indices.
    :rtype: List[Dict]

    :Example:

    ::

        >>> import ransac
        >>> import open3d as o3d
        >>> import numpy as np
        >>> import random
        >>> import open3d as o3d
        >>> dataset = o3d.data.OfficePointClouds()
        >>> pcds_offices = []
        >>> for pcd_path in dataset.paths:
        >>>     pcds_offices.append(o3d.io.read_point_cloud(pcd_path))
        >>> office_pcd = pcds_offices[0]
        >>> pcd_points = np.asarray(office_pcd.points)
        >>> threshold = 0.1
        >>> num_iterations = 20
        >>> dict_results = ransac.get_ransac_plane_results(pcd_points, threshold, num_iterations, seed = 42)
        >>> dict_results
        {'best_plane': array([-0.17535096,  0.45186984, -2.44615646,  5.69205427]),
        'number_inliers': 153798,
        'indices_inliers': array([     0,      1,      2, ..., 248476, 248477, 248478])}
        >>> fitting_data = coreransac.get_fitting_data_from_list_planes(pcd_points, [dict_results["best_plane"]], threshold)
        >>> fitting_data
        [{'plane': array([-0.17535096,  0.45186984, -2.44615646,  5.69205427]),
        'number_inliers': 153798,
        'indices_inliers': array([     0,      1,      2, ..., 248476, 248477, 248478])}]
    ''' 
    list_fitting_data = []
    for plane in list_planes:
        how_many_in_plane, indices_inliers = get_how_many_below_threshold_between_plane_and_points_and_their_indices(points, plane, threshold)
        list_fitting_data.append({"plane": plane, "number_inliers": how_many_in_plane, "indices_inliers": indices_inliers})
    return list_fitting_data

def get_best_fitting_data_from_list_planes(points: np.ndarray, list_planes: List[np.ndarray], threshold: float) -> Dict:
    '''
    Returns a dictionary containing the **plane parameters**, **number of inliers**, and **their indices** for the 
    **best plane** in a **given list**.
    
    It takes an array of points, a list of plane equations, and a threshold for defining inliers. 
    The function evaluates each plane for its fitting quality and returns a dictionary with the parameters of the plane 
    that has the highest inlier count, along with the count and indices of these inliers. This function expects a 
    **collection of points**, a **list of plane equations**, and a **threshold** for inlier determination. It returns a 
    dictionary with the plane's parameters, the count of inliers, and the indices of these inliers. The keys of the dictionary 
    are **"plane"**, **"number_inliers"**, and **"indices_inliers"**, respectively. The type of the values of the dictionary are **np.ndarray**, **int**, and **np.ndarray**, respectively.

    :param points: The collection of points to fit the plane to.
    :type points: np.ndarray
    :param list_planes: The list of planes to fit to the points.
    :type list_planes: List[np.ndarray]
    :param threshold: The maximum distance from a point to the plane for it to be considered an inlier.
    :type threshold: float
    :return: A dictionary containing the plane parameters, number of inliers, and their indices.
    :rtype: Dict

    :Example:

    ::

        >>> import ransac
        >>> import open3d as o3d
        >>> import numpy as np
        >>> import random
        >>> import open3d as o3d
        >>> dataset = o3d.data.OfficePointClouds()
        >>> pcds_offices = []
        >>> for pcd_path in dataset.paths:
        >>>     pcds_offices.append(o3d.io.read_point_cloud(pcd_path))
        >>> office_pcd = pcds_offices[0]
        >>> pcd_points = np.asarray(office_pcd.points)
        >>> threshold = 0.1
        >>> num_iterations = 20
        >>> dict_results = ransac.get_ransac_plane_results(pcd_points, threshold, num_iterations, seed = 42)
        >>> dict_results
        {'best_plane': array([-0.17535096,  0.45186984, -2.44615646,  5.69205427]),
        'number_inliers': 153798,
        'indices_inliers': array([     0,      1,      2, ..., 248476, 248477, 248478])}
        >>> fitting_data = coreransac.get_fitting_data_from_list_planes(pcd_points, [dict_results["best_plane"]], threshold)
        >>> fitting_data
        [{'plane': array([-0.17535096,  0.45186984, -2.44615646,  5.69205427]),
        'number_inliers': 153798,
        'indices_inliers': array([     0,      1,      2, ..., 248476, 248477, 248478])}]
        >>> best_fitting_data = coreransac.get_best_fitting_data_from_list_planes(pcd_points, [dict_results["best_plane"]], threshold)
        >>> best_fitting_data
        {'plane': array([-0.17535096,  0.45186984, -2.44615646,  5.69205427]),
        'number_inliers': 153798,
        'indices_inliers': array([     0,      1,      2, ..., 248476, 248477, 248478])}
    '''
    fitting_data = get_fitting_data_from_list_planes(points, list_planes, threshold)
    best_fitting_data = max(fitting_data, key=lambda fitting_data: fitting_data["number_inliers"])
    return best_fitting_data
