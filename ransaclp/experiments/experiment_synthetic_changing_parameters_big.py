import os
import open3d as o3d
import numpy as np
import time
import ransaclp
import rsaitehu_ransac as ransac
import matplotlib.pyplot as plt

# Global threshold for segmentation.
common_threshold = 0.02

def create_random_point_cloud(num_plane_points, num_cube_points, plane_color, cube_color):
    """
    Creates an Open3D point cloud composed of:
      1. Points on a planar surface near Z=0 with X and Y in [-1, 1],
         with added Gaussian noise on Z (mean=0, std=0.01).
      2. Random points uniformly distributed inside a cube with bounds X, Y, Z in [-2, 2].
    """
    # Plane points.
    plane_xy = np.random.uniform(-1, 1, size=(num_plane_points, 2))
    plane_z = np.random.normal(0, 0.01, size=(num_plane_points, 1))
    plane_points = np.hstack((plane_xy, plane_z))
    
    # Cube (outlier) points.
    cube_points = np.random.uniform(-2, 2, size=(num_cube_points, 3))
    
    # Combine points and colors.
    points = np.vstack((plane_points, cube_points))
    plane_colors = np.tile(np.array(plane_color), (num_plane_points, 1))
    cube_colors = np.tile(np.array(cube_color), (num_cube_points, 1))
    colors = np.vstack((plane_colors, cube_colors))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def ransaclp_segmentation(pcd, threshold=0.01, iterations=1000,
                          percentage_chosen_lines=0.2, use_cuda=True,
                          percentage_chosen_planes=0.05, seed=42):
    """
    Performs plane segmentation using the ransaclp module.
    Returns the plane model, inliers, and elapsed time.
    """
    # Warm-up call.
    plane_model, inliers = ransaclp.segment_plane(
        pcd,
        distance_threshold=threshold,
        num_iterations=iterations,
        percentage_chosen_lines=percentage_chosen_lines,
        use_cuda=use_cuda,
        percentage_chosen_planes=percentage_chosen_planes,
        seed=seed
    )
    start_time = time.time()
    plane_model, inliers = ransaclp.segment_plane(
        pcd,
        distance_threshold=threshold,
        num_iterations=iterations,
        percentage_chosen_lines=percentage_chosen_lines,
        use_cuda=use_cuda,
        percentage_chosen_planes=percentage_chosen_planes,
        seed=seed
    )
    elapsed = time.time() - start_time
    return plane_model, inliers, elapsed

def run_parameter_grid_experiments_and_save_results():
    """
    Varies percentage_chosen_lines and percentage_chosen_planes over a grid,
    runs ransaclp segmentation 10 times for each combination, computes the average
    running time and inlier count, and saves the results in /tmp/parameters_big.
    """
    out_dir = "/tmp/parameters_big"
    os.makedirs(out_dir, exist_ok=True)
    
    num_plane_points = 10000
    num_cube_points = num_plane_points * 5  # Fixed ratio 5.
    
    # Define parameter ranges.
    pct_lines_vals = np.linspace(0.05, 0.3, 20)      # 20 values from 0.05 to 0.3.
    pct_planes_vals = np.linspace(0.01, 0.1, 20)       # 20 values from 0.01 to 0.1.
    
    # Arrays to store results.
    mean_times = np.zeros((len(pct_planes_vals), len(pct_lines_vals)))
    mean_inliers = np.zeros((len(pct_planes_vals), len(pct_lines_vals)))
    
    num_repetitions = 10
    iterations = 600
    threshold = common_threshold
    seed = 42
    
    # Loop over the grid.
    for i, pct_planes in enumerate(pct_planes_vals):
        for j, pct_lines in enumerate(pct_lines_vals):
            times = []
            inlier_counts = []
            for rep in range(num_repetitions):
                pcd = create_random_point_cloud(num_plane_points, num_cube_points,
                                                plane_color=(1, 0, 0),
                                                cube_color=(0, 0, 1))
                try:
                    _, inliers, elapsed = ransaclp_segmentation(
                        pcd, threshold=threshold, iterations=iterations,
                        percentage_chosen_lines=pct_lines,
                        use_cuda=True,
                        percentage_chosen_planes=pct_planes,
                        seed=seed
                    )
                    times.append(elapsed)
                    inlier_counts.append(len(inliers))
                except Exception as e:
                    print(f"Error for lines {pct_lines:.3f} and planes {pct_planes:.3f}: {e}")
                    times.append(0)
                    inlier_counts.append(0)
            mean_times[i, j] = np.mean(times)
            mean_inliers[i, j] = np.mean(inlier_counts)
            print(f"Lines: {pct_lines:.3f}, Planes: {pct_planes:.3f} -> Mean Time: {mean_times[i, j]:.4f} s, Mean Inliers: {mean_inliers[i, j]:.0f}")
    
    # Save the computed data.
    save_path = os.path.join(out_dir, "parameter_grid_results.npz")
    np.savez(save_path,
             pct_lines_vals=pct_lines_vals,
             pct_planes_vals=pct_planes_vals,
             mean_times=mean_times,
             mean_inliers=mean_inliers)
    print(f"Parameter grid results saved to {save_path}")

def create_heatmap_from_saved_results():
    """
    Loads the experiment results from /tmp/parameters_big and creates heatmaps
    for the mean running time and mean inlier counts. The plot image is saved in the same directory.
    """
    out_dir = "/tmp/parameters_big"
    results_path = os.path.join(out_dir, "parameter_grid_results.npz")
    
    if not os.path.exists(results_path):
        print("Results file not found. Please run run_parameter_grid_experiments_and_save_results() first.")
        return
    
    # Load saved results.
    data = np.load(results_path)
    pct_lines_vals = data["pct_lines_vals"]
    pct_planes_vals = data["pct_planes_vals"]
    mean_times = data["mean_times"]
    mean_inliers = data["mean_inliers"]
    
    # Create heatmaps.
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap for running time.
    im0 = axs[0].imshow(mean_times, aspect='auto', origin='lower',
                        extent=[pct_lines_vals[0], pct_lines_vals[-1],
                                pct_planes_vals[0], pct_planes_vals[-1]],
                        cmap='viridis')
    axs[0].set_title("Mean Running Time (s)")
    axs[0].set_xlabel("Percentage Chosen Lines")
    axs[0].set_ylabel("Percentage Chosen Planes")
    fig.colorbar(im0, ax=axs[0])
    
    # Heatmap for inlier counts.
    im1 = axs[1].imshow(mean_inliers, aspect='auto', origin='lower',
                        extent=[pct_lines_vals[0], pct_lines_vals[-1],
                                pct_planes_vals[0], pct_planes_vals[-1]],
                        cmap='viridis')
    axs[1].set_title("Mean Inlier Count")
    axs[1].set_xlabel("Percentage Chosen Lines")
    axs[1].set_ylabel("Percentage Chosen Planes")
    fig.colorbar(im1, ax=axs[1])
    
    plt.tight_layout()
    plot_save_path = os.path.join(out_dir, 'ransaclp_param_grid_heatmaps.eps')
    plt.savefig(plot_save_path, format='eps', dpi=300)
    print(f"Heatmap plot saved to {plot_save_path}")
    # plt.show()

if __name__ == "__main__":
    # Uncomment the following line to run experiments and save results:
    # run_parameter_grid_experiments_and_save_results()
    
    # Uncomment the following line to load results and create the heatmap:
    create_heatmap_from_saved_results()

