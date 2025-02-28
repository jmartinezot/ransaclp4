import os
import open3d as o3d
import numpy as np
import time
import ransaclp
import rsaitehu_ransac as ransac
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

plt.rcParams.update({'font.size': 16})

# Global threshold for standard RANSAC
common_threshold = 0.02

def create_random_point_cloud(num_plane_points, num_cube_points, plane_color, cube_color):
    """
    Creates an Open3D point cloud composed of:
      1. Points on a planar surface near Z=0 with X and Y in [-1, 1],
         with added Gaussian noise in the Z axis (mean=0, std=0.01).
      2. Random points uniformly distributed inside a cube with bounds:
         X, Y, Z in [-2, 2].
    """
    # Generate plane points: x and y uniformly distributed in [-1, 1]
    plane_xy = np.random.uniform(-1, 1, size=(num_plane_points, 2))
    plane_z = np.random.normal(0, 0.01, size=(num_plane_points, 1))
    plane_points = np.hstack((plane_xy, plane_z))
    
    # Generate cube points: each coordinate uniformly distributed in [-2, 2]
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
    Performs plane segmentation using the ransaclp module,
    measures the time taken, and returns the plane model, inliers, and elapsed time.
    """
    # Initial segmentation call to warm up (if necessary)
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

def ransac_segmentation(pcd):
    """
    Performs Open3D's standard RANSAC plane segmentation on the point cloud,
    measures the time taken, and returns the plane model, inliers, and elapsed time.
    """
    start_time = time.time()
    plane_model, inliers = pcd.segment_plane(distance_threshold=common_threshold,
                                             ransac_n=3,
                                             num_iterations=957)
    elapsed = time.time() - start_time
    return plane_model, inliers, elapsed

def run_experiments_and_save_results():
    """
    Varies the ratio of outliers (cube points) to inliers (plane points),
    runs segmentation algorithms 10 times for each configuration,
    and saves the mean segmentation times and inlier counts in /tmp/percentage_outliers_big.
    """
    # Ensure the output directory exists.
    out_dir = "/tmp/percentage_outliers_big"
    os.makedirs(out_dir, exist_ok=True)
    
    num_plane_points = 100000
    outlier_ratios = np.linspace(0, 5.0, 200)
    
    standard_times = []
    standard_inliers = []
    ransaclp_times = []
    ransaclp_inliers = []
    
    threshold = common_threshold
    iterations = 600
    percentage_chosen_lines = 0.2
    percentage_chosen_planes = 0.05
    seed = 42
    num_repetitions = 10  # Number of iterations for each configuration.
    
    for ratio in outlier_ratios:
        num_cube_points = int(num_plane_points * ratio)
        std_times_temp = []
        std_inliers_temp = []
        ransaclp_times_temp = []
        ransaclp_inliers_temp = []
        
        for _ in range(num_repetitions):
            pcd = create_random_point_cloud(num_plane_points, num_cube_points,
                                            plane_color=(1, 0, 0),
                                            cube_color=(0, 0, 1))
            
            # Standard RANSAC segmentation.
            _, inliers_std, time_std = ransac_segmentation(pcd)
            std_times_temp.append(time_std)
            std_inliers_temp.append(len(inliers_std))
            
            # RANSACLP segmentation.
            try:
                _, inliers_lp, time_lp = ransaclp_segmentation(
                    pcd, threshold=threshold, iterations=iterations,
                    percentage_chosen_lines=percentage_chosen_lines,
                    use_cuda=True,
                    percentage_chosen_planes=percentage_chosen_planes,
                    seed=seed
                )
            except Exception as e:
                print("Error in RANSACLP segmentation:", e)
                inliers_lp = []
                time_lp = 0
            ransaclp_times_temp.append(time_lp)
            ransaclp_inliers_temp.append(len(inliers_lp))
        
        # Compute mean values over the repetitions.
        mean_std_time = np.mean(std_times_temp)
        mean_std_inliers = np.mean(std_inliers_temp)
        mean_lp_time = np.mean(ransaclp_times_temp)
        mean_lp_inliers = np.mean(ransaclp_inliers_temp)
        
        standard_times.append(mean_std_time)
        standard_inliers.append(mean_std_inliers)
        ransaclp_times.append(mean_lp_time)
        ransaclp_inliers.append(mean_lp_inliers)
    
    # Save the computed data.
    save_path = os.path.join(out_dir, "experiment_results.npz")
    np.savez(save_path,
             outlier_ratios=outlier_ratios,
             standard_times=standard_times,
             standard_inliers=standard_inliers,
             ransaclp_times=ransaclp_times,
             ransaclp_inliers=ransaclp_inliers)
    print(f"Experiment results saved to {save_path}")

def create_plot_from_saved_results():
    """
    Loads the experiment results from /tmp/percentage_outliers_big and creates
    two plots: one for segmentation time vs outlier ratio, and one for number of inliers vs outlier ratio.
    The resulting plot image is saved in the same directory.
    """
    out_dir = "/tmp/percentage_outliers_big"
    results_path = os.path.join(out_dir, "experiment_results.npz")
    
    if not os.path.exists(results_path):
        print("Results file not found. Please run run_experiments_and_save_results() first.")
        return
    
    # Load the saved results.
    data = np.load(results_path)
    outlier_ratios = data["outlier_ratios"]
    standard_times = data["standard_times"]
    standard_inliers = data["standard_inliers"]
    ransaclp_times = data["ransaclp_times"]
    ransaclp_inliers = data["ransaclp_inliers"]

    # Use Seaborn's colorblind palette.
    cb_palette = sns.color_palette("colorblind")
    color_standard = cb_palette[0]  # For Standard RANSAC
    color_ransaclp = cb_palette[1]  # For RANSACLP
    
    # Create the plot.
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot mean segmentation time vs outlier ratio.
    axs[0].scatter(outlier_ratios, standard_times, marker='o', label="Standard RANSAC", color=color_standard)
    axs[0].scatter(outlier_ratios, ransaclp_times, marker='o', label="RANSAC-LP4", color=color_ransaclp)
    axs[0].set_xlabel("Outlier Ratio (cube points / plane points)")
    axs[0].set_ylabel("Mean Segmentation Time (seconds)")
    axs[0].set_title("Segmentation Time vs Outlier Ratio")
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot mean number of inliers vs outlier ratio.
    axs[1].scatter(outlier_ratios, standard_inliers, marker='o', label="Standard RANSAC", color=color_standard)
    axs[1].scatter(outlier_ratios, ransaclp_inliers, marker='o', label="RANSAC-LP4", color=color_ransaclp)
    axs[1].set_xlabel("Outlier Ratio (cube points / plane points)")
    axs[1].set_ylabel("Mean Number of Inliers")
    axs[1].set_title("Inliers Count vs Outlier Ratio")
    axs[1].legend()
    axs[1].grid(True)

    axs[1].yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    
    plt.tight_layout()
    plot_save_path = os.path.join(out_dir, 'synthetic_changing_percentage_outiers_big.eps')
    plt.savefig(plot_save_path, format='eps', dpi=300)
    print(f"Plot saved to {plot_save_path}")
    # plt.show()

if __name__ == "__main__":
    # Uncomment one of the lines below to run the desired function.
    # First, run the experiments and save the results:
    # run_experiments_and_save_results()
    
    # Then, create the plot from the saved results:
    create_plot_from_saved_results()
