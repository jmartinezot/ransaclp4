from . import ransac
from . import ransaccuda
from . import pointcloud # ok
from . import geometry as geom # ok
from . import sampling # ok
import open3d as o3d
import numpy as np
from typing import Callable, List, Tuple, Dict
from numba import cuda
from numba import njit, prange

def segment_plane(pcd: o3d.geometry.PointCloud, distance_threshold: float, num_iterations: int, 
                  percentage_chosen_lines: float = 0.2, percentage_chosen_planes: float  = 0.05, 
                  use_cuda: bool = False, verbosity_level: int = 0, seed: int = None):
    '''
    Segments a plane from a point cloud.

    :param pcd: The point cloud.
    :type pcd: o3d.geometry.PointCloud
    :param distance_threshold: The distance threshold.
    :type distance_threshold: float
    :param num_iterations: The number of iterations.
    :type num_iterations: int
    :return: The plane model and the inliers.
    :rtype: Tuple[np.ndarray, np.ndarray]

    :Example:

    ::

        >>> import open3d as o3d
        >>> import ransaclp
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> pcd = o3d.io.read_point_cloud(office_filename)
        >>> plane_model, inliers = ransaclp.segment_plane(pcd, distance_threshold=0.01, num_iterations=1000, percentage_chosen_lines = 0.2, percentage_chosen_planes = 0.05, seed = 42)
        >>> plane_model
        array([-0.07208637,  0.20357587, -0.97640177,  2.25114528])
        >>> number_inliers = len(inliers)
        >>> inlier_cloud = pcd.select_by_index(inliers)
        >>> inlier_cloud.paint_uniform_color([1, 0, 0])
        >>> o3d.visualization.draw_geometries([pcd, inlier_cloud], window_name="RANSACLP Inliers:  " + str(number_inliers))
    '''
    np_points = np.asarray(pcd.points)
    ransaclp_best_data, ransaclp_full_data = get_ransaclp_data_from_np_points(np_points, ransac_iterations = num_iterations, 
                                                           threshold = distance_threshold,
                                                           use_cuda = use_cuda,
                                                           percentage_chosen_lines = percentage_chosen_lines,
                                                           percentage_chosen_planes = percentage_chosen_planes,
                                                           verbosity_level = verbosity_level, 
                                                           inherited_verbose_string = "",
                                                           seed = seed)
    ransaclp_plane = ransaclp_best_data["plane"]
    indices_inliers = ransaclp_best_data["indices_inliers"]
    return ransaclp_plane, indices_inliers

def get_ransac_data_from_np_points(np_points: np.ndarray, ransac_iterator: Callable, ransac_iterations: int = 100, threshold: float = 0.1, 
                                  verbosity_level: int = 0, inherited_verbose_string: str = "",
                                  seed: int = None) -> Dict:
    '''
    Gets the ransac data from a file. 
    
    The ransac data is a dictionary with the following keys:
    - ransac_iterations_results: the results of the ransac iterations. It is a list, where each element is a dictionary with the following keys:
        - current_random_points: the current random points chosen to estimate the line
        - current_line: the current line estimated from the random points. Right now it is a numpy array with two points, the same as the current_random_points
        - threshold: the maximum distance from the line to consider a point as an inlier
        - number_inliers: the number of inliers
        - indices_inliers: the indices of the inliers in the point cloud
    - ransac_best_iteration_results: the best iteration results

    :param np_points: The point cloud data.
    :type np_points: np.ndarray
    :param ransac_iterator: The ransac iterator.
    :type ransac_iterator: Callable, function that takes the point cloud data and returns the ransac data.
    :param ransac_iterations: The number of ransac iterations.
    :type ransac_iterations: int
    :param threshold: The threshold.
    :type threshold: float
    :param audit_cloud: Whether to audit the cloud.
    :type audit_cloud: bool
    :param verbosity_level: The verbosity level.
    :type verbosity_level: int
    :param inherited_verbose_string: The inherited verbose string.
    :type inherited_verbose_string: str
    :param seed: The seed.
    :type seed: int
    :return: The ransac data.
    :rtype: dict

    :Example:

    ::

        >>> import open3d as o3d
        >>> import numpy as np
        >>> import ransaclp
        >>> import ransac
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> ransac_iterator = ransac.get_ransac_line_iteration_results
        >>> ransac_iterations = 200
        >>> threshold = 0.02
        >>> seed = 42
        >>> pcd = o3d.io.read_point_cloud(office_filename)
        >>> np_points = np.asarray(pcd.points)
        >>> ransac_data = ransaclp.get_ransac_data_from_np_points(np_points, ransac_iterator = ransac_iterator, 
                                                           ransac_iterations = ransac_iterations, 
                                                           threshold = threshold, seed = seed)
        >>> iterations_results = ransac_data["ransac_iterations_results"]
        >>> len(iterations_results)
        200
        >>> first_iteration_results = iterations_results[0]
        >>> first_iteration_results["current_random_points"]
        array([[1.88671875, 1.96484375, 1.91060746],
        [3.18359375, 1.89562058, 2.45703125]])
        >>> first_iteration_results["current_line"]
        array([[1.88671875, 1.96484375, 1.91060746],
        [3.18359375, 1.89562058, 2.45703125]])
        >>> first_iteration_results["threshold"]
        0.02
        >>> first_iteration_results["number_inliers"]
        203
        >>> first_iteration_results["indices_inliers"][:5]
        array([ 58884,  59966,  60516,  61070, 138037])
        >>> best_iteration_results = ransac_data["ransac_best_iteration_results"]
        >>> best_iteration_results["current_random_points"]
        array([[1.85546875, 2.67396379, 2.08203125],
        [0.85546875, 2.57869077, 2.52734375]])
        >>> best_iteration_results["current_line"]
        array([[1.85546875, 2.67396379, 2.08203125],
        [0.85546875, 2.57869077, 2.52734375]])
        >>> best_iteration_results["threshold"]
        0.02
        >>> best_iteration_results["number_inliers"]
        2128
        >>> best_iteration_results["indices_inliers"][:5]
        array([24335, 25743, 25746, 25897, 26405])
    '''
    if seed is not None:
        np.random.seed(seed)

    dict_full_results = {}
    dict_full_results["ransac_iterations_results"] = []

    max_number_inliers = 0
    best_iteration_results = None
    current_iteration = 0
    for i in range(ransac_iterations):
        current_iteration += 1
        message = f"{inherited_verbose_string} Fitting line to pointcloud: Iteration {current_iteration}/{ransac_iterations}"
        if (verbosity_level == 1 and current_iteration % 10 == 0) or verbosity_level >= 2:
            print(message)
        else:
            dict_iteration_results = ransac_iterator(np_points, threshold, len(np_points))
        if dict_iteration_results["number_inliers"] > max_number_inliers:
            max_number_inliers = dict_iteration_results["number_inliers"]
            best_iteration_results = dict_iteration_results
        dict_full_results["ransac_iterations_results"].append(dict_iteration_results)
    dict_full_results["ransac_best_iteration_results"] = best_iteration_results
    return dict_full_results

#@profile
def get_ransac_data_from_np_points_cuda(np_points: np.ndarray, ransac_iterator: Callable, max_threads_per_block, ransac_iterations: int = 100, threshold: float = 0.1, 
                                  verbosity_level: int = 0, inherited_verbose_string: str = "",
                                  seed: int = None) -> Dict:
    '''
    Gets the ransac data from a file. 
    
    The ransac data is a dictionary with the following keys:
    - ransac_iterations_results: the results of the ransac iterations. It is a list, where each element is a dictionary with the following keys:
        - current_random_points: the current random points chosen to estimate the line
        - current_line: the current line estimated from the random points. Right now it is a numpy array with two points, the same as the current_random_points
        - threshold: the maximum distance from the line to consider a point as an inlier
        - number_inliers: the number of inliers
        - indices_inliers: the indices of the inliers in the point cloud
    - ransac_best_iteration_results: the best iteration results

    :param np_points: The point cloud data.
    :type np_points: np.ndarray
    :param ransac_iterator: The ransac iterator.
    :type ransac_iterator: Callable, function that takes the point cloud data and returns the ransac data.
    :param ransac_iterations: The number of ransac iterations.
    :type ransac_iterations: int
    :param threshold: The threshold.
    :type threshold: float
    :param audit_cloud: Whether to audit the cloud.
    :type audit_cloud: bool
    :param verbosity_level: The verbosity level.
    :type verbosity_level: int
    :param inherited_verbose_string: The inherited verbose string.
    :type inherited_verbose_string: str
    :param seed: The seed.
    :type seed: int
    :return: The ransac data.
    :rtype: dict

    :Example:

    ::

        >>> import open3d as o3d
        >>> import ransaclp
        >>> import ransaccuda
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> ransac_iterator = ransaccuda.get_ransac_line_iteration_results_cuda
        >>> ransac_iterations = 200
        >>> threshold = 0.02
        >>> seed = 42
        >>> pcd = o3d.io.read_point_cloud(office_filename)
        >>> np_points = np.asarray(pcd.points)
        >>> ransac_data = ransaclp.get_ransac_data_from_filename(np_points, ransac_iterator = ransac_iterator, 
                                                           ransac_iterations = ransac_iterations, 
                                                           threshold = threshold, audit_cloud=True, seed = seed)
        >>> iterations_results = ransac_data["ransac_iterations_results"]
        >>> len(iterations_results)
        200
        >>> first_iteration_results = iterations_results[0]
        >>> first_iteration_results["current_random_points"]
        array([[1.88671875, 1.96484375, 1.91060746],
        [3.18359375, 1.89562058, 2.45703125]])
        >>> first_iteration_results["current_line"]
        array([[1.88671875, 1.96484375, 1.91060746],
        [3.18359375, 1.89562058, 2.45703125]])
        >>> first_iteration_results["threshold"]
        0.02
        >>> first_iteration_results["number_inliers"]
        203
        >>> first_iteration_results["indices_inliers"][:5]
        array([ 58884,  59966,  60516,  61070, 138037])
        >>> best_iteration_results = ransac_data["ransac_best_iteration_results"]
        >>> best_iteration_results["current_random_points"]
        array([[1.85546875, 2.67396379, 2.08203125],
        [0.85546875, 2.57869077, 2.52734375]])
        >>> best_iteration_results["current_line"]
        array([[1.85546875, 2.67396379, 2.08203125],
        [0.85546875, 2.57869077, 2.52734375]])
        >>> best_iteration_results["threshold"]
        0.02
        >>> best_iteration_results["number_inliers"]
        2128
        >>> best_iteration_results["indices_inliers"][:5]
        array([24335, 25743, 25746, 25897, 26405])
    '''
    if seed is not None:
        np.random.seed(seed)

    dict_full_results = {}
    dict_full_results["ransac_iterations_results"] = []

    max_number_inliers = 0
    best_iteration_results = None
    # current_iteration = 0
    # Extract x, y, z coordinates from np_points
    points_x = np_points[:, 0]
    points_y = np_points[:, 1]
    points_z = np_points[:, 2]
    # Convert points_x, points_y, and points_z to contiguous arrays
    d_points_x = cuda.to_device(np.ascontiguousarray(points_x))
    d_points_y = cuda.to_device(np.ascontiguousarray(points_y))
    d_points_z = cuda.to_device(np.ascontiguousarray(points_z))
    random_numbers_pairs = sampling.sampling_np_arrays_from_enumerable(np_points, cardinality_of_np_arrays=2, 
                                                                       number_of_np_arrays=ransac_iterations, 
                                                                       num_source_elems=len(np_points), seed=seed)

    # Preallocate device memory for the result once (outside the loop)
    host_results = np.zeros(ransac_iterations, dtype=np.int32)
    d_results = cuda.to_device(host_results)

    # Precompute slices for each iteration.
    result_slices = [d_results[i:i+1] for i in range(ransac_iterations)]

    # Choose a batch size (adjust as needed)
    batch_size = 100

    for batch_start in range(0, ransac_iterations, batch_size):
        # Launch a batch of kernel calls asynchronously
        for i in range(batch_start, min(batch_start + batch_size, ransac_iterations)):
            current_iteration = i + 1
            message = f"{inherited_verbose_string} Fitting line to pointcloud: Iteration {current_iteration}/{ransac_iterations}"
            if (verbosity_level == 1 and current_iteration % 10 == 0) or verbosity_level >= 2:
                print(message)

            random_points = random_numbers_pairs[i]
            # Each kernel call writes its output into d_results[i:i+1] (a one-element slice)
            iteration_result = ransac_iterator(np_points,
                            d_points_x, d_points_y, d_points_z,
                            threshold, random_points,
                            result_slices[i], max_threads_per_block)
            # The returned dictionary has placeholder for "number_inliers".
            dict_full_results["ransac_iterations_results"].append(iteration_result)
        
        # Synchronize once after each batch.
        cuda.synchronize()

    # Copy the full results array from device to host in one transfer.
    host_results = d_results.copy_to_host()

    # (Optional) You could update dict_full_results using host_results if needed.
    # For example, if you want to include the number of inliers per iteration:
    for i in range(ransac_iterations):
        dict_full_results["ransac_iterations_results"][i]["number_inliers"] = int(host_results[i])

    # Now, determine the best iteration using the updated results.
    max_inliers = -1
    for iteration_result in dict_full_results["ransac_iterations_results"]:
        if iteration_result["number_inliers"] > max_inliers:
            max_inliers = iteration_result["number_inliers"]
            best_iteration_results = iteration_result
    
    dict_full_results["ransac_best_iteration_results"] = best_iteration_results
    return dict_full_results
 
    '''
    for i in range(ransac_iterations):
        current_iteration += 1
        message = f"{inherited_verbose_string} Fitting line to pointcloud: Iteration {current_iteration}/{ransac_iterations}"
        if (verbosity_level == 1 and current_iteration % 10 == 0) or verbosity_level >= 2:
            print(message)

        # Reset the result on device if needed, e.g., by copying a zero array
        d_result.copy_to_device(np.array([0], dtype=np.int32))

        random_points = random_numbers_pairs[i]
        dict_iteration_results = ransac_iterator(np_points, d_points_x, d_points_y, d_points_z, threshold, random_points, d_result)

        if dict_iteration_results["number_inliers"] > max_number_inliers:
            max_number_inliers = dict_iteration_results["number_inliers"]
            best_iteration_results = dict_iteration_results
        dict_full_results["ransac_iterations_results"].append(dict_iteration_results)
    dict_full_results["ransac_best_iteration_results"] = best_iteration_results
    return dict_full_results
    '''


def get_ransac_data_from_filename(filename: str, ransac_iterator: Callable, ransac_iterations: int = 100, threshold: float = 0.1, 
                                  audit_cloud: bool = False, verbosity_level: int = 0, inherited_verbose_string: str = "",
                                  seed: int = None) -> Dict:
    '''
    Gets the ransac data from a file. 
    
    The ransac data is a dictionary with the following keys:
    - filename: the filename
    - audit_before_sanitizing: the audit of the point cloud before sanitizing; it is a dictionary with the following keys:
        - number_pcd_points: the number of points in the point cloud
        - has_normals: whether the point cloud has normals
        - has_colors: whether the point cloud has colors
        - is_empty: whether the point cloud is empty
        - max_x: the maximum x coordinate
        - min_x: the minimum x coordinate
        - max_y: the maximum y coordinate
        - min_y: the minimum y coordinate
        - max_z: the maximum z coordinate
        - min_z: the minimum z coordinate
        - all_points_finite: whether all the points are finite
        - all_points_unique: whether all the points are unique
    - audit_after_sanitizing: the audit of the point cloud after sanitizing
    - number_pcd_points: the number of points in the point cloud
    - ransac_iterations_results: the results of the ransac iterations. It is a list, where each element is a dictionary with the following keys:
        - current_random_points: the current random points chosen to estimate the line
        - current_line: the current line estimated from the random points. Right now it is a numpy array with two points, the same as the current_random_points
        - threshold: the maximum distance from the line to consider a point as an inlier
        - number_inliers: the number of inliers
        - indices_inliers: the indices of the inliers in the point cloud
    - ransac_best_iteration_results: the best iteration results

    :param filename: The filename.
    :type filename: str
    :param ransac_iterator: The ransac iterator.
    :type ransac_iterator: Callable, function that takes the point cloud data and returns the ransac data.
    :param ransac_iterations: The number of ransac iterations.
    :type ransac_iterations: int
    :param threshold: The threshold.
    :type threshold: float
    :param audit_cloud: Whether to audit the cloud.
    :type audit_cloud: bool
    :param verbosity_level: The verbosity level.
    :type verbosity_level: int
    :param inherited_verbose_string: The inherited verbose string.
    :type inherited_verbose_string: str
    :param seed: The seed.
    :type seed: int
    :return: The ransac data.
    :rtype: dict

    :Example:

    ::

        >>> import open3d as o3d
        >>> import ransaclp
        >>> import ransac
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> ransac_iterator = ransac.get_ransac_line_iteration_results
        >>> ransac_iterations = 200
        >>> threshold = 0.02
        >>> seed = 42
        >>> ransac_data = ransaclp.get_ransac_data_from_filename(office_filename, ransac_iterator = ransac_iterator, 
                                                           ransac_iterations = ransac_iterations, 
                                                           threshold = threshold, audit_cloud=True, seed = seed)
        >>> ransac_data["filename"]
        '/home/user/open3d_data/extract/OfficePointClouds/cloud_bin_0.ply'
        >>> ransac_data["audit_before_sanitizing"]
        {'number_pcd_points': 276871,
        'has_normals': True,
        'has_colors': True,
        'is_empty': False,
        'max_x': 3.5121824741363525,
        'min_x': 0.00390625,
        'max_y': 2.80859375,
        'min_y': 0.47265625,
        'max_z': 2.7512423992156982,
        'min_z': 0.94921875,
        'all_points_finite': True,
        'all_points_unique': True}
        >>> ransac_data["audit_after_sanitizing"]
        {'number_pcd_points': 276871,
        'has_normals': True,
        'has_colors': True,
        'is_empty': False,
        'max_x': 3.5121824741363525,
        'min_x': 0.00390625,
        'max_y': 2.80859375,
        'min_y': 0.47265625,
        'max_z': 2.7512423992156982,
        'min_z': 0.94921875,
        'all_points_finite': True,
        'all_points_unique': True}
        >>> ransac_data["number_pcd_points"]
        276871
        >>> iterations_results = ransac_data["ransac_iterations_results"]
        >>> len(iterations_results)
        200
        >>> first_iteration_results = iterations_results[0]
        >>> first_iteration_results["current_random_points"]
        array([[1.88671875, 1.96484375, 1.91060746],
        [3.18359375, 1.89562058, 2.45703125]])
        >>> first_iteration_results["current_line"]
        array([[1.88671875, 1.96484375, 1.91060746],
        [3.18359375, 1.89562058, 2.45703125]])
        >>> first_iteration_results["threshold"]
        0.02
        >>> first_iteration_results["number_inliers"]
        203
        >>> first_iteration_results["indices_inliers"][:5]
        array([ 58884,  59966,  60516,  61070, 138037])
        >>> best_iteration_results = ransac_data["ransac_best_iteration_results"]
        >>> best_iteration_results["current_random_points"]
        array([[1.85546875, 2.67396379, 2.08203125],
        [0.85546875, 2.57869077, 2.52734375]])
        >>> best_iteration_results["current_line"]
        array([[1.85546875, 2.67396379, 2.08203125],
        [0.85546875, 2.57869077, 2.52734375]])
        >>> best_iteration_results["threshold"]
        0.02
        >>> best_iteration_results["number_inliers"]
        2128
        >>> best_iteration_results["indices_inliers"][:5]
        array([24335, 25743, 25746, 25897, 26405])
    '''
    if seed is not None:
        np.random.seed(seed)

    dict_full_results = {}
    dict_full_results["filename"] = filename

    pcd = o3d.io.read_point_cloud(filename)
    if audit_cloud:
        audit_before_sanitizing = pointcloud.pointcloud_audit(pcd)
        dict_full_results["audit_before_sanitizing"] = audit_before_sanitizing
        pcd = pointcloud.pointcloud_sanitize(pcd)
        audit_after_sanitizing = pointcloud.pointcloud_audit(pcd)
        dict_full_results["audit_after_sanitizing"] = audit_after_sanitizing
    
    number_pcd_points = len(pcd.points)
    dict_full_results["number_pcd_points"] = number_pcd_points
    np_points = np.asarray(pcd.points)

    dict_results_from_np_points = get_ransac_data_from_np_points(np_points, ransac_iterator = ransac_iterator, 
                                                                 ransac_iterations = ransac_iterations, 
                                                                 threshold = threshold, 
                                                                 verbosity_level = verbosity_level, 
                                                                 inherited_verbose_string = inherited_verbose_string,
                                                                 seed = seed)

    dict_full_results.update(dict_results_from_np_points)
    return dict_full_results

# create a function that returns all the current_line along with their number_inliers

def get_lines_and_number_inliers_from_ransac_data_from_file(ransac_data_from_file):
    pair_lines_number_inliers = []
    for dict_iteration_results in ransac_data_from_file["ransac_iterations_results"]:
        pair_lines_number_inliers.append((dict_iteration_results["current_line"], dict_iteration_results["number_inliers"]))
    return pair_lines_number_inliers

# order the pairs by number_inliers

def get_lines_and_number_inliers_ordered_by_number_inliers(ransac_data: Dict) -> List[Tuple[np.ndarray, int]]:
    '''
    Gets the lines and number of inliers ordered by number of inliers from ransac data previously extracted.

    :param ransac_data: The ransac data.
    :type ransac_data: Dict
    :return: The list of pairs of lines and number of inliers.
    :rtype: List[Tuple[np.ndarray, int]]

    :Example:

    ::

        >>> import open3d as o3d
        >>> import ransaclp
        >>> import ransac
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> ransac_iterator = ransac.get_ransac_line_iteration_results
        >>> ransac_iterations = 200
        >>> threshold = 0.02
        >>> seed = 42
        >>> ransac_data = ransaclp.get_ransac_data_from_filename(office_filename, ransac_iterator = ransac_iterator,
                                                                 ransac_iterations = ransac_iterations,
                                                                 threshold = threshold, audit_cloud=True, seed = seed)
        >>> pair_lines_number_inliers_ordered = ransaclp.get_lines_and_number_inliers_ordered_by_number_inliers(ransac_data)
        >>> pair_lines_number_inliers_ordered[:5]
        [(array([[1.85546875, 2.67396379, 2.08203125],
                [0.85546875, 2.57869077, 2.52734375]]),
        2128),
        (array([[0.19140625, 2.10900044, 2.62890625],
                [1.03637648, 1.11328125, 2.43359375]]),
        1988),
        (array([[3.12890625, 2.17578125, 2.52116251],
                [1.27803993, 0.67578125, 2.33203125]]),
        1960),
        (array([[0.31464097, 0.90234375, 2.46484375],
                [2.99609375, 1.31640625, 2.34281754]]),
        1953),
        (array([[1.06640625, 0.86328125, 2.37622762],
                [1.03515625, 1.08984375, 2.42773271]]),
        1925)]


    '''
    pair_lines_number_inliers = get_lines_and_number_inliers_from_ransac_data_from_file(ransac_data)
    pair_lines_number_inliers_ordered = sorted(pair_lines_number_inliers, key=lambda pair_line_number_inliers: pair_line_number_inliers[1], reverse=True)
    return pair_lines_number_inliers_ordered

@njit
def compute_plane_from_four_points(p0, p1, p2, p3):
    # Compute the centroid manually
    cx = (p0[0] + p1[0] + p2[0] + p3[0]) * 0.25
    cy = (p0[1] + p1[1] + p2[1] + p3[1]) * 0.25
    cz = (p0[2] + p1[2] + p2[2] + p3[2]) * 0.25

    # Compute shifted points
    s0 = (p0[0] - cx, p0[1] - cy, p0[2] - cz)
    s1 = (p1[0] - cx, p1[1] - cy, p1[2] - cz)
    s2 = (p2[0] - cx, p2[1] - cy, p2[2] - cz)
    s3 = (p3[0] - cx, p3[1] - cy, p3[2] - cz)

    # Compute covariance matrix (symmetric 3x3)
    cov = np.zeros((3, 3))
    for s in (s0, s1, s2, s3):
        cov[0, 0] += s[0] * s[0]
        cov[0, 1] += s[0] * s[1]
        cov[0, 2] += s[0] * s[2]
        cov[1, 1] += s[1] * s[1]
        cov[1, 2] += s[1] * s[2]
        cov[2, 2] += s[2] * s[2]
    cov[1, 0] = cov[0, 1]
    cov[2, 0] = cov[0, 2]
    cov[2, 1] = cov[1, 2]

    # Use eigh for symmetric matrices; eigenvalues are returned in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Normal is the eigenvector corresponding to the smallest eigenvalue.
    normal = eigenvectors[:, 0]

    # Compute plane offset d: for plane defined as n·x + d = 0, d = -n·centroid
    d = -(normal[0] * cx + normal[1] * cy + normal[2] * cz)
    return normal, d

@njit(parallel=True)
def compute_all_planes(pair_lines, number_best, results_sse, results_plane):
    # Assume pair_lines is a 2D array-like container of shape (n, 2, 3)
    # where each row has two points (each point is 3D).
    # results_sse and results_plane are preallocated arrays.
    k = 0
    for i in prange(number_best):
        for j in range(i + 1, number_best):
            # Retrieve the endpoints from the two lines.
            # Each line is a 2x3 array.
            line1 = pair_lines[i]
            line2 = pair_lines[j]
            p0 = line1[0]
            p1 = line1[1]
            p2 = line2[0]
            p3 = line2[1]

            normal, d = compute_plane_from_four_points(p0, p1, p2, p3)
            
            # Compute SSE for the 4 points
            sse = 0.0
            for p in (p0, p1, p2, p3):
                err = normal[0] * p[0] + normal[1] * p[1] + normal[2] * p[2] + d
                if err < 0:
                    err = -err
                sse += err * err

            results_sse[k] = sse
            # Store plane parameters as [normal_x, normal_y, normal_z, d]
            results_plane[k, 0] = normal[0]
            results_plane[k, 1] = normal[1]
            results_plane[k, 2] = normal[2]
            results_plane[k, 3] = d
            k += 1

    return k  # number of computed planes

# Example usage:
# Let's say you have 'pair_lines_number_inliers', where each element is (line, inlier_count)
# and line is a 2x3 array. First, extract the best lines into an array.
def prepare_pair_lines(pair_lines_number_inliers, number_best):
    # We assume that pair_lines_number_inliers is already sorted.
    pair_lines = np.empty((number_best, 2, 3), dtype=np.float64)
    for i in range(number_best):
        pair_lines[i] = pair_lines_number_inliers[i][0]
    return pair_lines

#@profile
def get_only_best_list_sse_plane(pair_lines_number_inliers: List[Tuple[np.ndarray, int]], 
                                      number_best: int = None, percentage_best_lines: float = 0.2, 
                                      percentage_best_planes: float = 0.05,
                                      already_ordered: bool = False, verbosity_level: int = 0, 
                                      inherited_verbose_string: str = "") -> List[Tuple[float, np.ndarray]]:
    # Determine how many candidates to process
    if number_best is None:
        number_best = int(len(pair_lines_number_inliers) * percentage_best_lines)
    
    # Optionally sort if not already sorted
    if not already_ordered:
        pair_lines_number_inliers = sorted(pair_lines_number_inliers, key=lambda x: x[1], reverse=True)
    
    # Extract candidate lines: shape (number_best, 2, 3)
    candidate_lines = np.array([item[0] for item in pair_lines_number_inliers[:number_best]])
    
    # Number of candidate pairs: we need all unique combinations i<j.
    n = candidate_lines.shape[0]
    # Get indices of the upper triangle (excluding diagonal)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # For each pair, get the endpoints
    # Each candidate line is 2x3; for pair (i,j), the four points are:
    #   p0, p1 from candidate_lines[i] and p2, p3 from candidate_lines[j]
    pts0 = candidate_lines[i_idx, 0, :]  # shape (n_pairs, 3)
    pts1 = candidate_lines[i_idx, 1, :]  # shape (n_pairs, 3)
    pts2 = candidate_lines[j_idx, 0, :]  # shape (n_pairs, 3)
    pts3 = candidate_lines[j_idx, 1, :]  # shape (n_pairs, 3)
    
    # Stack the four points for each pair to get an array of shape (n_pairs, 4, 3)
    points = np.stack([pts0, pts1, pts2, pts3], axis=1)
    
    # Compute the centroid for each set of 4 points (shape: (n_pairs, 3))
    centroids = points.mean(axis=1)
    
    # Compute shifted points: subtract the centroid from each point
    shifted_points = points - centroids[:, None, :]
    
    # Perform a stacked SVD: one SVD per (4, 3) matrix.
    # Using full_matrices=False gives U: (n_pairs, 4, 3), s: (n_pairs, 3), Vh: (n_pairs, 3, 3)
    _, _, Vh = np.linalg.svd(shifted_points, full_matrices=False)
    
    # The plane normal corresponds to the singular vector associated with the smallest singular value.
    # Vh has shape (n_pairs, 3, 3); we take the last row from each SVD.
    normals = Vh[:, 2, :]  # shape (n_pairs, 3)
    
    # Compute plane offset d for each pair: d = - normal dot centroid
    d = -np.einsum('ij,ij->i', normals, centroids)  # shape (n_pairs,)
    
    # Compute errors for each of the four original points:
    # For each pair, compute dot product for each of the 4 points:
    errors = np.abs(np.einsum('ik,ijk->ij', normals, points) + d[:, None])
    
    # Sum of squared errors (SSE) for each pair
    sse = np.sum(errors ** 2, axis=1)
    
    # Build plane arrays: each plane is [normal_x, normal_y, normal_z, d]
    planes = np.hstack([normals, d[:, None]])  # shape (n_pairs, 4)

    # Determine how many best planes to choose.
    k = int(percentage_best_planes * len(sse))
    if k < 1:
        k = 1
    
    # Get indices of the k smallest SSE values (order not guaranteed).
    part_idx = np.argpartition(sse, k)[:k]
    # Fully sort these k elements.
    best_idx = part_idx[np.argsort(sse[part_idx])]
    
    # Build the final list of (sse, plane) tuples.
    list_sse_plane = [(float(sse[i]), planes[i]) for i in best_idx]
    
    if verbosity_level > 0:
        print(f"{inherited_verbose_string}Selected top {k} planes out of {len(sse)} candidate pairs.")
    
    return list_sse_plane

def get_ordered_list_sse_plane(pair_lines_number_inliers:List[Tuple[np.ndarray, int]], number_best: int = None, percentage_best: float = 0.2, 
                               already_ordered: bool = False, verbosity_level: int = 0, inherited_verbose_string: str = "") -> List[Tuple[float, np.ndarray]]:
    '''
    Gets a list of the sse (sum of squared errors) of the planes built from the best pairs of lines, along with the planes. 
    The best pairs of lines are the ones with the highest number of inliers. 
    The number of planes returned is either the number_best or the percentage_best of the total number of planes.

    :param pair_lines_number_inliers: The list of pairs of lines and number of inliers.
    :type pair_lines_number_inliers: List[Tuple[np.ndarray, int]]
    :param number_best: The number of best planes to consider.
    :type number_best: int
    :param percentage_best: The percentage of best planes to consider.
    :type percentage_best: float
    :param already_ordered: Whether the pairs are already ordered by number of inliers.
    :type already_ordered: bool
    :param verbosity_level: The verbosity level.
    :type verbosity_level: int
    :param inherited_verbose_string: The inherited verbose string.
    :type inherited_verbose_string: str
    :return: The list of sse and planes ordered by sse.
    :rtype: List[Tuple[float, np.ndarray]]

    :Example:

    ::

        >>> import open3d as o3d
        >>> import ransaclp
        >>> import ransac
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> ransac_iterator = coreransac.get_ransac_line_iteration_results
        >>> ransac_iterations = 200
        >>> threshold = 0.02
        >>> seed = 42
        >>> ransac_data = ransaclp.get_ransac_data_from_filename(office_filename, ransac_iterator = ransac_iterator,
                                                                    ransac_iterations = ransac_iterations,
                                                                    threshold = threshold, audit_cloud=True, seed = seed)
        >>> pair_lines_number_inliers = ransaclp.get_lines_and_number_inliers_from_ransac_data_from_file(ransac_data)
        >>> list_sse_plane = ransaclp.get_ordered_list_sse_plane(pair_lines_number_inliers, percentage_best = 0.2)
        >>> list_sse_plane[:5]
        [(3.133004699296308e-09,
        array([ 0.02115254,  0.23488194, -0.97179373,  2.07212126])),
        (4.072922344774246e-09,
        array([-0.0583794 ,  0.16947913, -0.98380317,  2.274669  ])),
        (5.597729648076989e-09,
        array([ 0.04009822, -0.15939724,  0.98639984, -2.26464593])),
        (1.6036620702538505e-08,
        array([ 0.04939934, -0.14154284,  0.98869881, -2.30467834])),
        (1.7597194644519755e-08,
        array([-0.02745968,  0.98751003,  0.15514477, -2.9031455 ]))]
    '''
    if number_best is None:
        number_best = int(len(pair_lines_number_inliers) * percentage_best)
    if not already_ordered:
        pair_lines_number_inliers = sorted(pair_lines_number_inliers, key=lambda x: x[1], reverse=True)
    list_sse_plane = []
    current_iteration = 0
    # compute the maximum number of iterations
    total_number_iterations = int(number_best * (number_best - 1) / 2)
    for i in range(number_best):
        for j in range(i+1, number_best):
            # compute the current number of iterations
            current_iteration += 1
            message = f"Estimating planes: Line pair {current_iteration}/{total_number_iterations}"
            if verbosity_level == 1 and current_iteration % 10 == 0 or verbosity_level >= 2:
                message = f"{inherited_verbose_string} {message}"
                print(message)
            line_1 = pair_lines_number_inliers[i][0]
            line_2 = pair_lines_number_inliers[j][0]
            plane, error = geom.get_best_plane_from_points_from_two_segments(line_1, line_2)
            list_sse_plane.append((error, plane))
    # order the list by sse
    list_sse_plane = sorted(list_sse_plane, key=lambda x: x[0])
    return list_sse_plane

def get_n_percentile_from_list_sse_plane(list_sse_plane: List[Tuple[float, np.ndarray]], percentile: int = 5) -> List[Tuple[float, np.ndarray]]:
    '''
    Gets the list of elements in which sse is below the n percentile from a list of sse and planes.
    The function looks for the best sse values, that are the lowest ones.

    :param list_sse_plane: The list of sse and planes.
    :type list_sse_plane: List[Tuple[float, np.ndarray]]
    :param percentile: The percentile.
    :type percentile: int
    :return: The n percentile.
    :rtype: List[Tuple[float, np.ndarray]]

    :Example:

    ::

        >>> import open3d as o3d
        >>> import ransaclp
        >>> import ransac
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> ransac_iterator = ransac.get_ransac_line_iteration_results
        >>> ransac_iterations = 200
        >>> threshold = 0.02
        >>> seed = 42
        >>> ransac_data = ransaclp.get_ransac_data_from_filename(office_filename, ransac_iterator = ransac_iterator,
                                                                    ransac_iterations = ransac_iterations,
                                                                    threshold = threshold, audit_cloud=True, seed = seed)
        >>> pair_lines_number_inliers = ransaclp.get_lines_and_number_inliers_from_ransac_data_from_file(ransac_data)
        >>> list_sse_plane = ransaclp.get_ordered_list_sse_plane(pair_lines_number_inliers, percentage_best = 0.2)
        >>> len(list_sse_plane)
        780
        >>> list_sse_plane[:5]
        [(3.133004699296308e-09,
        array([ 0.02115254,  0.23488194, -0.97179373,  2.07212126])),
        (4.072922344774246e-09,
        array([-0.0583794 ,  0.16947913, -0.98380317,  2.274669  ])),
        (5.597729648076989e-09,
        array([ 0.04009822, -0.15939724,  0.98639984, -2.26464593])),
        (1.6036620702538505e-08,
        array([ 0.04939934, -0.14154284,  0.98869881, -2.30467834])),
        (1.7597194644519755e-08,
        array([-0.02745968,  0.98751003,  0.15514477, -2.9031455 ]))]
        >>> list_sse_plane_05 = ransaclp.get_n_percentile_from_list_sse_plane(list_sse_plane, percentile = 5)
        >>> len(list_sse_plane_05)
        39
        >>> list_sse_plane_05[:5]
        [(3.133004699296308e-09,
        array([ 0.02115254,  0.23488194, -0.97179373,  2.07212126])),
        (4.072922344774246e-09,
        array([-0.0583794 ,  0.16947913, -0.98380317,  2.274669  ])),
        (5.597729648076989e-09,
        array([ 0.04009822, -0.15939724,  0.98639984, -2.26464593])),
        (1.6036620702538505e-08,
        array([ 0.04939934, -0.14154284,  0.98869881, -2.30467834])),
        (1.7597194644519755e-08,
        array([-0.02745968,  0.98751003,  0.15514477, -2.9031455 ]))]

    '''
    list_sse, list_plane = zip(*list_sse_plane)  # Unpack the list of tuples
    percentile_threshold = np.percentile(list_sse, percentile)
    filtered_pairs = [(sse, plane) for sse, plane in zip(list_sse, list_plane) if sse <= percentile_threshold]
    return filtered_pairs

def get_ransaclp_data_from_filename(filename: str, ransac_iterations: int = 100, threshold: float = 0.1, audit_cloud: bool = False, 
                                    percentage_chosen_lines: float = 0.2, percentage_chosen_planes: float = 0.05,
                                    verbosity_level: int = 0, inherited_verbose_string: str = "", seed: int = None) -> Dict:
    '''
    Gets the ransaclp data from a file.
    
    :param filename: The filename.
    :type filename: str
    :param ransac_iterations: The number of ransac iterations.
    :type ransac_iterations: int
    :param threshold: The threshold.
    :type threshold: float
    :param audit_cloud: Whether to audit the cloud.
    :type audit_cloud: bool
    :param verbosity_level: The verbosity level.
    :type verbosity_level: int
    :param inherited_verbose_string: The inherited verbose string.
    :type inherited_verbose_string: str
    :param seed: The seed.
    :type seed: int
    :return: The ransaclp data.
    :rtype: dict

    :Example:

    ::

        >>> import open3d as o3d
        >>> import ransaclp
        >>> import ransac
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> ransac_iterations = 200
        >>> threshold = 0.02
        >>> seed = 42
        >>> ransaclp_data = ransaclp.get_ransaclp_data_from_filename(office_filename, 
                                                                    ransac_iterations = ransac_iterations, 
                                                                    threshold = threshold, audit_cloud=True, seed = seed)
        >>> ransaclp_data
        {'plane': array([-0.07208637,  0.20357587, -0.97640177,  2.25114528]),
        'number_inliers': 83788,
        'indices_inliers': array([     7,      8,     10, ..., 248476, 248477, 248478])}
        >>> pcd = o3d.io.read_point_cloud(office_filename)
        >>> number_inliers = ransaclp_data["number_inliers"]
        >>> indices_inliers = ransaclp_data["indices_inliers"]
        >>> inlier_cloud = pcd.select_by_index(indices_inliers)
        >>> inlier_cloud.paint_uniform_color([1, 0, 0])
        >>> o3d.visualization.draw_geometries([pcd, inlier_cloud], window_name="RANSACLP Inliers:  " + str(number_inliers))

    '''
    if seed is not None:
        np.random.seed(seed)
    ransac_iterator = ransac.get_ransac_line_iteration_results
    ransac_data = get_ransac_data_from_filename(filename, ransac_iterator = ransac_iterator,
                                                ransac_iterations = ransac_iterations,
                                                threshold = threshold,
                                                audit_cloud=audit_cloud, verbosity_level = verbosity_level, 
                                                inherited_verbose_string=inherited_verbose_string, seed = seed)
    pcd = o3d.io.read_point_cloud(ransac_data["filename"])
    np_points = np.asarray(pcd.points)

    pair_lines_number_inliers = get_lines_and_number_inliers_from_ransac_data_from_file(ransac_data)
    ordered_list_sse_plane = get_ordered_list_sse_plane(pair_lines_number_inliers, percentage_best = percentage_chosen_lines, verbosity_level=verbosity_level,
                                                        inherited_verbose_string=inherited_verbose_string)
    percentile = percentage_chosen_planes * 100,
    list_sse_plane_05 = get_n_percentile_from_list_sse_plane(ordered_list_sse_plane, percentile = percentile)
    list_good_planes = [sse_plane[1] for sse_plane in list_sse_plane_05]
    results_from_best_plane = ransac.get_best_fitting_data_from_list_planes(np_points, list_good_planes, threshold)
    return results_from_best_plane

#@profile
def get_ransaclp_data_from_np_points(np_points: np.ndarray, max_threads_per_block: int = 1024, ransac_iterations: int = 100, threshold: float = 0.1,
                                     use_cuda: bool = False, percentage_chosen_lines: float = 0.2, percentage_chosen_planes: float = 0.05,
                                    verbosity_level: int = 0, inherited_verbose_string: str = "", seed: int = None) -> Dict:
    '''
    Gets the ransaclp data from a file.
    
    :param np_points: The numpy array of points.
    :type np_points: np.ndarray
    :param ransac_iterations: The number of ransac iterations.
    :type ransac_iterations: int
    :param threshold: The threshold.
    :type threshold: float
    :param cuda: Wheter cuda is used.
    :type cuda: bool
    :param verbosity_level: The verbosity level.
    :type verbosity_level: int
    :param inherited_verbose_string: The inherited verbose string.
    :type inherited_verbose_string: str
    :param seed: The seed.
    :type seed: int
    :return: The ransaclp data.
    :rtype: dict

    :Example:

    ::

        >>> import open3d as o3d
        >>> import numpy as np
        >>> import ransaclp
        >>> import ransac
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> ransac_iterations = 200
        >>> threshold = 0.02
        >>> seed = 42
        >>> pcd = o3d.io.read_point_cloud(office_filename)
        >>> np_points = np.asarray(pcd.points)
        >>> ransaclp_data = ransaclp.get_ransaclp_data_from_np_points(np_points, 
                                                                    ransac_iterations = ransac_iterations, 
                                                                    threshold = threshold, seed = seed)
        >>> ransaclp_data
        {'plane': array([-0.07208637,  0.20357587, -0.97640177,  2.25114528]),
        'number_inliers': 83788,
        'indices_inliers': array([     7,      8,     10, ..., 248476, 248477, 248478])}
        >>> pcd = o3d.io.read_point_cloud(office_filename)
        >>> number_inliers = ransaclp_data["number_inliers"]
        >>> indices_inliers = ransaclp_data["indices_inliers"]
        >>> inlier_cloud = pcd.select_by_index(indices_inliers)
        >>> inlier_cloud.paint_uniform_color([1, 0, 0])
        >>> o3d.visualization.draw_geometries([pcd, inlier_cloud], window_name="RANSACLP Inliers:  " + str(number_inliers))

    '''
    if seed is not None:
        np.random.seed(seed)
    if use_cuda:
        ransac_iterator = ransaccuda.get_ransac_line_iteration_results_cuda
        ransac_data = get_ransac_data_from_np_points_cuda(np_points, ransac_iterator = ransac_iterator,
                                            max_threads_per_block = max_threads_per_block,
                                            ransac_iterations = ransac_iterations,
                                            threshold = threshold,
                                            verbosity_level = verbosity_level, 
                                            inherited_verbose_string=inherited_verbose_string, seed = seed)
    else:
        ransac_iterator = ransac.get_ransac_line_iteration_results
        ransac_data = get_ransac_data_from_np_points(np_points, ransac_iterator = ransac_iterator,
                                                    ransac_iterations = ransac_iterations,
                                                    threshold = threshold,
                                                    verbosity_level = verbosity_level, 
                                                    inherited_verbose_string=inherited_verbose_string, seed = seed)

    pair_lines_number_inliers = get_lines_and_number_inliers_from_ransac_data_from_file(ransac_data)
    only_best_list_sse_plane = get_only_best_list_sse_plane(pair_lines_number_inliers, 
                                                        percentage_best_lines = percentage_chosen_lines,
                                                        percentage_best_planes=percentage_chosen_planes, 
                                                        verbosity_level=verbosity_level,
                                                        inherited_verbose_string=inherited_verbose_string)
    list_good_planes = [sse_plane[1] for sse_plane in only_best_list_sse_plane]
    print("Number of good planes:", len(list_good_planes))
    print("First good plane:", list_good_planes[0])
    if use_cuda: 
        results_from_best_plane = ransaccuda.get_best_fitting_data_from_list_planes_cuda(np_points, list_good_planes, threshold)
    else:
        results_from_best_plane = ransac.get_best_fitting_data_from_list_planes(np_points, list_good_planes, threshold)
    return results_from_best_plane, ransac_data
