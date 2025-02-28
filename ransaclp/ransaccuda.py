'''
This module contains functions related to RANSAC and CUDA parallel processing.
'''
from numba import cuda, int32, float32
import math
from . import sampling
import numpy as np
from time import time
from typing import Tuple, List, Dict

@cuda.jit(fastmath=True)
def __get_how_many_and_which_below_threshold_kernel(points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray,
                                         a: float, b: float, c: float, d: float,
                                         optimized_threshold: float, point_indices: np.ndarray) -> None:
    """
    Computes the number of points that are below a threshold distance from a plane using CUDA parallel processing.
    
    :param points_x: The x-coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: The y-coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: The z-coordinates of the points.
    :type points_z: np.ndarray
    :param a: The first coefficient of the plane equation.
    :type a: float
    :param b: The second coefficient of the plane equation.
    :type b: float
    :param c: The third coefficient of the plane equation.
    :type c: float
    :param d: The fourth coefficient of the plane equation.
    :type d: float
    :param optimized_threshold: The threshold distance from the plane.
    :type optimized_threshold: float
    :param point_indices: The array of indices representing the points that are below the threshold.
    :type point_indices: np.ndarray
    """
    i = cuda.grid(1)
    if i < points_x.shape[0]:
        dist = math.fabs(a * points_x[i] + b * points_y[i] + c * points_z[i] + d)
        if dist <= optimized_threshold:
            point_indices[i] = 1

@cuda.jit(fastmath=True)
def __get_how_many_below_threshold_kernel(points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray,
                                         a: float, b: float, c: float, d: float,
                                         optimized_threshold: float, result: int) -> None:
    """
    Computes the number of points that are below a threshold distance from a plane using CUDA parallel processing.
    
    :param points_x: The x-coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: The y-coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: The z-coordinates of the points.
    :type points_z: np.ndarray
    :param a: The first coefficient of the plane equation.
    :type a: float
    :param b: The second coefficient of the plane equation.
    :type b: float
    :param c: The third coefficient of the plane equation.
    :type c: float
    :param d: The fourth coefficient of the plane equation.
    :type d: float
    :param optimized_threshold: The threshold distance from the plane.
    :type optimized_threshold: float
    :param point_indices: The array of indices representing the points that are below the threshold.
    :type point_indices: np.ndarray
    """
    i = cuda.grid(1) # to compute the index of the current thread
    if i < points_x.shape[0]:
        dist = math.fabs(a * points_x[i] + b * points_y[i] + c * points_z[i] + d)
        if dist <= optimized_threshold:
            cuda.atomic.add(result, 0, 1)

@cuda.jit(fastmath=True)
def __get_how_many_line_below_threshold_kernel_OLD(points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray,
                                         line_two_points: np.ndarray,
                                         threshold: float, result: int) -> None:
    """
    Computes the number of points that are below a threshold distance from a plane using CUDA parallel processing.
    
    :param points_x: The x-coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: The y-coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: The z-coordinates of the points.
    :type points_z: np.ndarray
    :param a: The first coefficient of the plane equation.
    :type a: float
    :param b: The second coefficient of the plane equation.
    :type b: float
    :param c: The third coefficient of the plane equation.
    :type c: float
    :param d: The fourth coefficient of the plane equation.
    :type d: float
    :param optimized_threshold: The threshold distance from the plane.
    :type optimized_threshold: float
    :param point_indices: The array of indices representing the points that are below the threshold.
    :type point_indices: np.ndarray
    """
    i = cuda.grid(1)
    if i < points_x.shape[0]:
        B = line_two_points[0]
        C = line_two_points[1]
        V_x = points_x[i] - B[0]
        V_y = points_y[i] - B[1]
        V_z = points_z[i] - B[2]
        V = (V_x, V_y, V_z)
        W_x = C[0] - B[0]
        W_y = C[1] - B[1]
        W_z = C[2] - B[2]
        W = (W_x, W_y, W_z)
        cross_product_x = V[1] * W[2] - V[2] * W[1]
        cross_product_y = V[2] * W[0] - V[0] * W[2]
        cross_product_z = V[0] * W[1] - V[1] * W[0]
        
        # magnitude_cross_product = math.sqrt(cross_product_x * cross_product_x + cross_product_y * cross_product_y + cross_product_z * cross_product_z)
        # magnitude_C_minus_B = math.sqrt((C[0] - B[0]) * (C[0] - B[0]) + (C[1] - B[1]) * (C[1] - B[1]) + (C[2] - B[2]) * (C[2] - B[2]))
        
        # dist = magnitude_cross_product / magnitude_C_minus_B
        # if dist <= threshold:
        #     cuda.atomic.add(result, 0, 1)

        # Compute squared norm of the cross product
        cross_norm_sq = (cross_product_x * cross_product_x +
                         cross_product_y * cross_product_y +
                         cross_product_z * cross_product_z)
        
        # Compute squared norm of (C - B)
        W_norm_sq = ((C[0] - B[0]) * (C[0] - B[0]) +
                     (C[1] - B[1]) * (C[1] - B[1]) +
                     (C[2] - B[2]) * (C[2] - B[2]))
        
        # Instead of sqrt, we compare squared values.
        # Condition: sqrt(cross_norm_sq / W_norm_sq) <= threshold  <==>
        #            cross_norm_sq / W_norm_sq <= threshold^2  <==>
        #            cross_norm_sq <= threshold^2 * W_norm_sq
        if cross_norm_sq <= threshold * threshold * W_norm_sq:
            cuda.atomic.add(result, 0, 1)

#@profile
def get_how_many_below_threshold_between_line_and_points_cuda_OLD(
    points: np.ndarray, d_points_x: np.ndarray, d_points_y: np.ndarray, d_points_z: np.ndarray,
    line_two_points: Tuple[Tuple[float, float, float], Tuple[float, float, float]], threshold: float) -> Tuple[int, List[int]]:

    """
    Computes the number of points that are below a threshold distance from a plane and their indices using CUDA parallel processing.
    
    :param points: The array of points in the format (x, y, z).
    :type points: np.ndarray
    :param points_x: The x-coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: The y-coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: The z-coordinates of the points.
    :type points_z: np.ndarray
    :param d_points_x: The x-coordinates of the points in device memory.
    :type d_points_x: np.ndarray
    :param d_points_y: The y-coordinates of the points in device memory.
    :type d_points_y: np.ndarray
    :param d_points_z: The z-coordinates of the points in device memory.
    :type d_points_z: np.ndarray
    :param plane: The coefficients of the plane equation.
    :type plane: Tuple[float, float, float, float]
    :param threshold: The threshold distance from the plane.
    :type threshold: float
    :return: The number of points below the threshold and their indices.
    :rtype: Tuple[int, List[int]]
    """
    t1 = time()
    num_points = points.shape[0]
    # Output variable to store the result
    result = np.array([0], dtype=np.int32)
    d_result = cuda.to_device(result)
    threadsperblock = 1024
    max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    threadsperblock = min(max_threads_per_block, threadsperblock)
    blockspergrid = math.ceil(num_points / threadsperblock)
    t2 = time()
    __get_how_many_line_below_threshold_kernel[blockspergrid, threadsperblock](d_points_x, d_points_y, d_points_z,
                                         line_two_points, threshold, d_result)
    t3 = time()
    # Copy the result back to the host
    cuda.synchronize()
    result = d_result.copy_to_host()[0]
    return result

@cuda.jit(fastmath=True)
def __get_how_many_line_below_threshold_kernel(points_x, points_y, points_z,
                                               B_x, B_y, B_z, W_x, W_y, W_z,
                                               threshold_factor, result):
    i = cuda.grid(1)
    if i < points_x.shape[0]:
        # Compute vector V from point B to the current point
        V_x = points_x[i] - B_x
        V_y = points_y[i] - B_y
        V_z = points_z[i] - B_z
        
        # Compute cross product between V and W
        cp_x = V_y * W_z - V_z * W_y
        cp_y = V_z * W_x - V_x * W_z
        cp_z = V_x * W_y - V_y * W_x
        
        # Compute the squared norm of the cross product
        cross_norm_sq = cp_x * cp_x + cp_y * cp_y + cp_z * cp_z
        
        # If the squared distance is within threshold, increment the counter
        if cross_norm_sq <= threshold_factor:
            cuda.atomic.add(result, 0, 1)

#@profile
def get_how_many_below_threshold_between_line_and_points_cuda(
    points: np.ndarray,
    d_points_x: np.ndarray, d_points_y: np.ndarray, d_points_z: np.ndarray,
    line_two_points: tuple,
    threshold: float,
    d_result, # Preallocated device slice (one-element array)
    max_threads_per_block: int # Pass the cached value here
    ) -> int:
    """
    Computes the number of points that are below a threshold distance from a line using CUDA parallel processing.
    
    :param points: The array of points (each point is (x, y, z)).
    :param d_points_x: The x-coordinates of the points in device memory.
    :param d_points_y: The y-coordinates of the points in device memory.
    :param d_points_z: The z-coordinates of the points in device memory.
    :param line_two_points: A tuple ((x1, y1, z1), (x2, y2, z2)) defining the line.
    :param threshold: The threshold distance.
    :return: The number of points below the threshold distance.
    """
    t1 = time()
    num_points = points.shape[0]
    
    # Allocate output result and move it to device memory
    # result = np.array([0], dtype=np.int32)
    # d_result = cuda.to_device(result)
    
    # Determine grid dimensions
    threadsperblock = 1024
    threadsperblock = min(max_threads_per_block, threadsperblock)
    blockspergrid = math.ceil(num_points / threadsperblock)
    t2 = time()
    
    # Precompute constants on the host
    B = line_two_points[0]
    C = line_two_points[1]
    # Convert B to a small numpy array for the kernel
    # B_arr = np.array(B, dtype=np.float32)
    
    # Compute vector W = C - B and its squared norm
    W_x = C[0] - B[0]
    W_y = C[1] - B[1]
    W_z = C[2] - B[2]
    W_norm_sq = W_x * W_x + W_y * W_y + W_z * W_z
    
    # Precompute the threshold factor (threshold^2 * ||W||^2)
    threshold_factor = threshold * threshold * W_norm_sq
    
    B_x, B_y, B_z = B  # unpack the tuple
    # Launch the kernel with precomputed constants
    __get_how_many_line_below_threshold_kernel[blockspergrid, threadsperblock](
        d_points_x, d_points_y, d_points_z,
        B_x, B_y, B_z, W_x, W_y, W_z,
        threshold_factor, d_result
    )
    t3 = time()
    
    # Wait for the kernel to finish and copy the result back
    # cuda.synchronize()
    #result = d_result.copy_to_host()[0]
    #return result

# OK
def get_how_many_and_which_below_threshold_between_plane_and_points_and_their_indices_cuda(
    points: np.ndarray, d_points_x: np.ndarray, d_points_y: np.ndarray, d_points_z: np.ndarray, 
    plane: Tuple[float, float, float, float], threshold: float) -> Tuple[int, List[int]]:
    """
    Computes the number of points that are below a threshold distance from a plane and their indices using CUDA parallel processing.
    
    :param points: The array of points in the format (x, y, z).
    :type points: np.ndarray
    :param points_x: The x-coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: The y-coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: The z-coordinates of the points.
    :type points_z: np.ndarray
    :param d_points_x: The x-coordinates of the points in device memory.
    :type d_points_x: np.ndarray
    :param d_points_y: The y-coordinates of the points in device memory.
    :type d_points_y: np.ndarray
    :param d_points_z: The z-coordinates of the points in device memory.
    :type d_points_z: np.ndarray
    :param plane: The coefficients of the plane equation.
    :type plane: Tuple[float, float, float, float]
    :param threshold: The threshold distance from the plane.
    :type threshold: float
    :return: The number of points below the threshold and their indices.
    :rtype: Tuple[int, List[int]]
    """
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]
    num_points = points.shape[0]
    point_indices = np.zeros(num_points, dtype=np.int32)
    optimized_threshold = threshold * math.sqrt(a * a + b * b + c * c)
    point_indices = np.empty(num_points, dtype=np.int64)
    # fill point_indices with -1
    point_indices[:] = -1
    threadsperblock = 512
    blockspergrid = math.ceil(num_points / threadsperblock)
    # point_indices = cuda.device_array(point_indices.shape, dtype=point_indices.dtype)
    d_point_indices = cuda.to_device(point_indices)
    __get_how_many_and_which_below_threshold_kernel[blockspergrid, threadsperblock](d_points_x, d_points_y, d_points_z, a, b, c, d, optimized_threshold, d_point_indices)
    point_indices = d_point_indices.copy_to_host()
    # get the count of point_indices that are not -1
    count = np.count_nonzero(point_indices != -1)
    # get the indices of the points that are not -1
    new_indices = np.where(point_indices != -1)
    new_indices = new_indices[0].tolist()
    return count, new_indices

# OK
def get_how_many_below_threshold_between_plane_and_points_cuda(
    points: np.ndarray, d_points_x: np.ndarray, d_points_y: np.ndarray, d_points_z: np.ndarray, 
    plane: Tuple[float, float, float, float], threshold: float) -> int:
    """
    Computes the number of points that are below a threshold distance from a plane and their indices using CUDA parallel processing.
    
    :param points: The array of points in the format (x, y, z).
    :type points: np.ndarray
    :param points_x: The x-coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: The y-coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: The z-coordinates of the points.
    :type points_z: np.ndarray
    :param d_points_x: The x-coordinates of the points in device memory.
    :type d_points_x: np.ndarray
    :param d_points_y: The y-coordinates of the points in device memory.
    :type d_points_y: np.ndarray
    :param d_points_z: The z-coordinates of the points in device memory.
    :type d_points_z: np.ndarray
    :param plane: The coefficients of the plane equation.
    :type plane: Tuple[float, float, float, float]
    :param threshold: The threshold distance from the plane.
    :type threshold: float
    :return: The number of points below the threshold and their indices.
    :rtype: Tuple[int, List[int]]
    """
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]
    num_points = points.shape[0]
    optimized_threshold = threshold * math.sqrt(a * a + b * b + c * c)
    # Output variable to store the result
    result = np.array([0], dtype=np.int32)
    d_result = cuda.to_device(result)
    threadsperblock = 1024
    max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    threadsperblock = min(max_threads_per_block, threadsperblock)
    blockspergrid = math.ceil(num_points / threadsperblock)
    __get_how_many_below_threshold_kernel[blockspergrid, threadsperblock](d_points_x, d_points_y, d_points_z, a, b, c, d, optimized_threshold, d_result)
    # Copy the result back to the host
    cuda.synchronize()
    result = d_result.copy_to_host()[0]
    return result

#@profile
def get_ransac_line_iteration_results_cuda(points: np.ndarray, 
                                       d_points_x: cuda.devicearray.DeviceNDArray, 
                                       d_points_y: cuda.devicearray.DeviceNDArray, 
                                       d_points_z: cuda.devicearray.DeviceNDArray, 
                                       threshold: float,
                                       random_points: np.ndarray,
                                       d_result_slice,  # Preallocated device slice for this iteration
                                       max_threads_per_block: int,
                                       ) -> dict:
    """
    Computes the number of inliers and the plane parameters for one iteration of the RANSAC algorithm using CUDA.

    :param points: Array of points.
    :type points: np.ndarray
    :param points_x: X coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: Y coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: Z coordinates of the points.
    :type points_z: np.ndarray
    :param d_points_x: Device array of X coordinates of the points.
    :type d_points_x: cuda.devicearray.DeviceNDArray
    :param d_points_y: Device array of Y coordinates of the points.
    :type d_points_y: cuda.devicearray.DeviceNDArray
    :param d_points_z: Device array of Z coordinates of the points.
    :type d_points_z: cuda.devicearray.DeviceNDArray
    :param num_points: Number of random points to select for each iteration.
    :type num_points: int
    :param threshold: Maximum distance to the plane.
    :type threshold: float
    :return: Dictionary with the plane parameters, the number of inliers, and the indices of the inliers.
    :rtype: dict
    """
    # this takes a lot of time
    if random_points is None:
        current_random_points = sampling.sampling_np_arrays_from_enumerable(points, cardinality_of_np_arrays=2, number_of_np_arrays=1, num_source_elems=len(points), seed=None)[0]
    else:
        current_random_points = random_points
    current_line = (tuple(current_random_points[0]), tuple(current_random_points[1]))
    get_how_many_below_threshold_between_line_and_points_cuda(points, d_points_x, d_points_y, d_points_z, current_line, threshold, d_result_slice, max_threads_per_block)
    # Return a dictionary with the computed parameters.
    # The actual number of inliers will be filled in later (after synchronization and copying).
    return {"current_line": current_random_points, "threshold": threshold, "number_inliers": None}

def get_fitting_data_from_list_planes_cuda_OLD(points: np.ndarray, list_planes: List[np.ndarray], threshold: float) -> List[Dict]:
    '''
    Returns the fitting data for each plane in the list of planes.

    :param points: The collection of points to fit the plane to.
    :type points: np.ndarray
    :param list_planes: The list of planes to fit to the points.
    :type list_planes: List[np.ndarray]
    :param threshold: The maximum distance from a point to the plane for it to be considered an inlier.
    :type threshold: float
    :return: A list of dictionaries containing the plane parameters, number of inliers, and their indices.
    :rtype: List[Dict]

    ''' 
    # Extract x, y, z coordinates from np_points
    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]
    # Convert points_x, points_y, and points_z to contiguous arrays
    d_points_x = np.ascontiguousarray(points_x)
    d_points_y = np.ascontiguousarray(points_y)
    d_points_z = np.ascontiguousarray(points_z)
    d_points_x = cuda.to_device(d_points_x)
    d_points_y = cuda.to_device(d_points_y)
    d_points_z = cuda.to_device(d_points_z)

    get_how_many_below_threshold_between_plane_and_points_cuda

    list_fitting_data = []
    for plane in list_planes:
        how_many_in_plane = get_how_many_below_threshold_between_plane_and_points_cuda(points, 
                                                                                    d_points_x=d_points_x,
                                                                                    d_points_y=d_points_y,
                                                                                    d_points_z=d_points_z,
                                                                                    plane = plane, 
                                                                                    threshold = threshold)
        list_fitting_data.append({"plane": plane, "number_inliers": how_many_in_plane})
    return list_fitting_data

@cuda.jit(fastmath=True)
def batched_plane_inlier_count(d_points_x, d_points_y, d_points_z, d_planes, threshold, d_results):
    """
    Each block processes one candidate plane.
    Each thread in the block processes a subset of points.
    """
    candidate_idx = cuda.blockIdx.x  # Each block is one candidate
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    num_points = d_points_x.shape[0]

    # Allocate shared memory for in-block reduction.
    # We assume block_size is not larger than 1024.
    shared_count = cuda.shared.array(1024, dtype=int32)
    shared_count[tid] = 0
    cuda.syncthreads()

    # Load the candidate plane parameters.
    # Each plane is defined by 4 coefficients: a, b, c, d.
    a = d_planes[candidate_idx, 0]
    b = d_planes[candidate_idx, 1]
    c = d_planes[candidate_idx, 2]
    d = d_planes[candidate_idx, 3]

    # Iterate over the points in a strided fashion.
    for i in range(tid, num_points, block_size):
        # Compute distance from point i to the plane.
        # (Optionally, adjust the threshold as in your original code.)
        dist = math.fabs(a * d_points_x[i] + b * d_points_y[i] + c * d_points_z[i] + d)
        if dist <= threshold:
            shared_count[tid] += 1
    cuda.syncthreads()

    # Reduction within the block to sum inlier counts.
    stride = block_size // 2
    while stride > 0:
        if tid < stride:
            shared_count[tid] += shared_count[tid + stride]
        cuda.syncthreads()
        stride //= 2

    # The first thread writes the candidate's result.
    if tid == 0:
        d_results[candidate_idx] = shared_count[0]

def get_fitting_data_from_list_planes_cuda(points: np.ndarray, list_planes: List[np.ndarray], threshold: float) -> List[Dict]:
    '''
    Returns the fitting data for each plane in the list of planes.
    
    :param points: The collection of points to fit the plane to.
    :type points: np.ndarray
    :param list_planes: The list of planes to fit to the points.
    :type list_planes: List[np.ndarray]
    :param threshold: The maximum distance from a point to the plane for it to be considered an inlier.
    :type threshold: float
    :return: A list of dictionaries containing the plane parameters and the number of inliers.
    :rtype: List[Dict]
    '''
    # Extract x, y, z coordinates from points.
    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]

    # Ensure the coordinate arrays are contiguous and move them to device memory.
    d_points_x = cuda.to_device(np.ascontiguousarray(points_x.astype(np.float32)))
    d_points_y = cuda.to_device(np.ascontiguousarray(points_y.astype(np.float32)))
    d_points_z = cuda.to_device(np.ascontiguousarray(points_z.astype(np.float32)))

    # Pack the candidate planes into a single 2D array.
    # Each plane is assumed to be a 1D array of 4 elements.
    candidate_planes = np.vstack(list_planes).astype(np.float32)  # shape: (num_candidates, 4)
    d_planes = cuda.to_device(candidate_planes)
    
    num_candidates = candidate_planes.shape[0]
    
    # Allocate a device array for results (one result per candidate).
    d_results = cuda.device_array(num_candidates, dtype=np.int32)
    
    # Choose a block size; adjust as appropriate.
    threadsperblock = 256
    blockspergrid = num_candidates  # One block per candidate plane.
    
    # Launch the batched kernel.
    batched_plane_inlier_count[blockspergrid, threadsperblock](d_points_x, d_points_y, d_points_z, d_planes, threshold, d_results)
    cuda.synchronize()
    
    # Copy results back to the host.
    results = d_results.copy_to_host()

    # Prepare the fitting data: for each candidate plane, record its parameters and its inlier count.
    list_fitting_data = []
    for i in range(num_candidates):
        list_fitting_data.append({
            "plane": list_planes[i],  # using the original candidate from the list
            "number_inliers": int(results[i])
        })
    return list_fitting_data

#@profile
def get_best_fitting_data_from_list_planes_cuda(points: np.ndarray, list_planes: List[np.ndarray], threshold: float) -> Dict:
    '''
    Returns the fitting data for the best plane in the list of planes.

    :param points: The collection of points to fit the plane to.
    :type points: np.ndarray
    :param list_planes: The list of planes to fit to the points.
    :type list_planes: List[np.ndarray]
    :param threshold: The maximum distance from a point to the plane for it to be considered an inlier.
    :type threshold: float
    :return: A dictionary containing the plane parameters, number of inliers, and their indices.
    :rtype: Dict

    '''
    fitting_data = get_fitting_data_from_list_planes_cuda(points, list_planes, threshold)
    best_fitting_data = max(fitting_data, key=lambda fitting_data: fitting_data["number_inliers"])
    plane = best_fitting_data["plane"]
    # Extract x, y, z coordinates from np_points
    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]
    # Convert points_x, points_y, and points_z to contiguous arrays
    d_points_x = np.ascontiguousarray(points_x)
    d_points_y = np.ascontiguousarray(points_y)
    d_points_z = np.ascontiguousarray(points_z)
    d_points_x = cuda.to_device(d_points_x)
    d_points_y = cuda.to_device(d_points_y)
    d_points_z = cuda.to_device(d_points_z)
    how_many_in_plane, indices_inliers = get_how_many_and_which_below_threshold_between_plane_and_points_and_their_indices_cuda(points, 
                                                                                                        d_points_x=d_points_x,
                                                                                                        d_points_y=d_points_y,
                                                                                                        d_points_z=d_points_z,
                                                                                                        plane = plane, 
                                                                                                        threshold = threshold)
    best_fitting_data["indices_inliers"] = indices_inliers
    return best_fitting_data