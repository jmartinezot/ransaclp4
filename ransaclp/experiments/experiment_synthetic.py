import open3d as o3d
import numpy as np
import time
import ransaclp
import rsaitehu_ransac as ransac

def create_random_point_cloud(num_plane_points, num_cube_points, plane_color, cube_color):
    """
    Creates an Open3D point cloud composed of:
      1. Points on a planar surface near Z=0 with X and Y in [-1, 1],
         with added Gaussian noise in the Z axis (mean=0, std=0.01).
      2. Random points uniformly distributed inside a cube with bounds:
         X, Y, Z in [-2, 2].
    
    Parameters:
        num_plane_points (int): Number of points to generate on the plane.
        num_cube_points (int): Number of points to generate inside the cube.
        plane_color (tuple or list of 3 floats): Color for the plane points (values in [0,1]).
        cube_color (tuple or list of 3 floats): Color for the cube points (values in [0,1]).
    
    Returns:
        o3d.geometry.PointCloud: The combined point cloud.
    """
    # Generate plane points: x and y uniformly distributed in [-1, 1]
    plane_xy = np.random.uniform(-1, 1, size=(num_plane_points, 2))
    # Z coordinate is centered at 0 with Gaussian noise (std=0.01)
    plane_z = np.random.normal(0, 0.01, size=(num_plane_points, 1))
    plane_points = np.hstack((plane_xy, plane_z))
    
    # Generate cube points: each coordinate uniformly distributed in [-2, 2]
    cube_points = np.random.uniform(-2, 2, size=(num_cube_points, 3))
    
    # Combine the two sets of points
    points = np.vstack((plane_points, cube_points))
    
    # Create color arrays for the points
    # Colors should be an array of shape (N, 3) with values between 0 and 1.
    plane_colors = np.tile(np.array(plane_color), (num_plane_points, 1))
    cube_colors = np.tile(np.array(cube_color), (num_cube_points, 1))
    colors = np.vstack((plane_colors, cube_colors))
    
    # Create the Open3D point cloud and assign points and colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def ransaclp_segmentation(pcd, threshold=0.01, iterations=1000,
                          percentage_chosen_lines=0.2,use_cuda=True,
                          percentage_chosen_planes=0.05,
                          seed=42):
    """
    Performs plane segmentation using the ransaclp module,
    measures the time taken, and returns the plane model, inliers, and elapsed time.
    """
    plane_model, inliers = ransaclp.segment_plane(
        pcd,
        distance_threshold=threshold,
        num_iterations=iterations,
        percentage_chosen_lines=percentage_chosen_lines,use_cuda=use_cuda,
        percentage_chosen_planes=percentage_chosen_planes,
        seed=seed
    )
    start_time = time.time()
    plane_model, inliers = ransaclp.segment_plane(
        pcd,
        distance_threshold=threshold,
        num_iterations=iterations,
        percentage_chosen_lines=percentage_chosen_lines,use_cuda=use_cuda,
        percentage_chosen_planes=percentage_chosen_planes,
        seed=seed
    )
    elapsed = time.time() - start_time
    print("RANSACLP segmentation took {:.4f} seconds".format(elapsed))
    print("RANSACLP plane model: {:.2f}x + {:.2f}y + {:.2f}z + {:.2f} = 0".format(*plane_model))
    print("Number of inliers (RANSACLP):", len(inliers))
    return plane_model, inliers, elapsed

def visualize_ransac(pcd, inliers, window_name):
    """
    Visualizes a segmentation result: the detected plane (red) and the remaining points (gray).
    """
    plane_cloud = pcd.select_by_index(inliers)
    plane_cloud.paint_uniform_color([1, 0, 0])  # red for the plane
    remaining_cloud = pcd.select_by_index(inliers, invert=True)
    remaining_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # light gray for the rest
    o3d.visualization.draw_geometries([plane_cloud, remaining_cloud],
                                      window_name=window_name)

def ransac_segmentation(pcd):
    """
    Performs Open3D's standard RANSAC plane segmentation on the point cloud,
    measures the time taken, and returns the plane model, inliers, and elapsed time.
    """
    plane_model, inliers = pcd.segment_plane(distance_threshold=common_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
    start_time = time.time()
    plane_model, inliers = pcd.segment_plane(distance_threshold=common_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
    elapsed = time.time() - start_time
    print("Standard RANSAC segmentation took {:.4f} seconds".format(elapsed))
    print("RANSAC plane model: {:.2f}x + {:.2f}y + {:.2f}z + {:.2f} = 0".format(*plane_model))
    print("Number of inliers (Standard RANSAC):", len(inliers))
    return plane_model, inliers, elapsed

# Example usage:
if __name__ == "__main__":
    common_threshold = 0.02

    # Create a point cloud with 1000 plane points (red) and 500 cube points (blue)
    pcd = create_random_point_cloud(10000, 40000, plane_color=(1, 0, 0), cube_color=(0, 0, 1))
    o3d.visualization.draw_geometries([pcd], window_name="Random Point Cloud")

    print("\n--- Running Standard RANSAC Segmentation ---")
    ransac_model, ransac_inliers, ransac_time = ransac_segmentation(pcd)

    print("\n--- Running RANSACLP Segmentation ---")
    # Set RANSACLP parameters.
    threshold = common_threshold
    iterations = 400
    percentage_chosen_lines = 0.2
    percentage_chosen_planes = 0.05
    seed = 42
    ransaclp_model, ransaclp_inliers, ransaclp_time = ransaclp_segmentation(
        pcd, threshold=threshold, iterations=iterations,
        percentage_chosen_lines=percentage_chosen_lines, use_cuda=True,
        percentage_chosen_planes=percentage_chosen_planes, seed=seed
    )

    # Summary of timings.
    print("\n--- Time Comparison ---")
    print("Standard RANSAC segmentation: {:.4f} seconds".format(ransac_time))
    print("RANSACLP segmentation:        {:.4f} seconds".format(ransaclp_time))

    # Optionally visualize results.
    visualize_ransac(pcd, ransac_inliers, window_name="Standard RANSAC Segmentation")
    visualize_ransac(pcd, ransaclp_inliers, window_name="RANSACLP Segmentation")
