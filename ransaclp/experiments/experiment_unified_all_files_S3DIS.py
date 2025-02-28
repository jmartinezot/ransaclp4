import os
import json
import glob
import hashlib
import numpy as np
import pandas as pd
import open3d as o3d
import ransaclp.ransaclpexperiments as experiments
from ransaclp.stats import (
    friedmanTest, friedmanPost, adjustShaffer, 
    to_heatmap_custom, to_image_custom
)
import matplotlib.pyplot as plt
import seaborn as sns
from numba import cuda

def make_json_serializable(obj):
    """Recursively convert non-JSON-serializable objects (e.g., NumPy arrays) to JSON-friendly types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    return obj

def unique_filename(filename, results_dir):
    safe_label = os.path.relpath(filename).replace(os.sep, "_")
    # Remove any leading dots or underscores so the filename isn't hidden
    safe_label = safe_label.lstrip('._')
    base, _ = os.path.splitext(safe_label)
    return os.path.join(results_dir, base + ".json")

def get_file_list(source_dir=None):
    """
    Return a list of file paths.
    
    If source_dir is provided, it searches for both .ply and .pcd files in that directory (recursively).
    Otherwise, it loads the default Open3D datasets (LivingRoomPointClouds and OfficePointClouds).
    """
    if source_dir is None:
        # Use Open3D's built-in datasets.
        dataset_lr = o3d.data.LivingRoomPointClouds()
        living_room_paths = dataset_lr.paths
        dataset_office = o3d.data.OfficePointClouds()
        office_paths = dataset_office.paths
        return living_room_paths + office_paths
    else:
        # Build file list from directory for both .ply and .pcd extensions.
        extensions = ["*.ply", "*.pcd"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(source_dir, "**", ext), recursive=True))
        return files

def run_experiments(results_dir, repetitions, iterations_list, threshold,
                    percentage_chosen_lines, percentage_chosen_planes, seed, source_dir=None):
    """
    Run experiments on a list of point cloud files.
    
    If source_dir is None, use Open3D's default datasets; otherwise, search the given directory.
    """
    os.makedirs(results_dir, exist_ok=True)
    file_list = get_file_list(source_dir)
    total_files = len(file_list)
    print(f"Found {total_files} files.")

    max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK

    for idx, filename in enumerate(file_list):
        verbose_str = f"File {idx + 1}/{total_files}: {os.path.basename(filename)}"
        print(f"Processing {verbose_str}")
        # result_dict = experiments.get_data_comparison_ransac_and_ransaclp(
        result_dict = experiments.get_data_comparison_ransac_ransaclp_and_planar_patches(
            filename=filename,
            repetitions=repetitions,
            iterations_list=iterations_list,
            threshold=threshold,
            percentage_chosen_lines=percentage_chosen_lines,
            percentage_chosen_planes=percentage_chosen_planes,
            max_threads_per_block=max_threads_per_block,
            cuda=True,
            verbosity_level=1,
            inherited_verbose_string=verbose_str,
            seed=seed
        )
        serializable_results = make_json_serializable(result_dict)
        filename_json = unique_filename(filename, results_dir)
        with open(filename_json, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"Saved results to {filename_json}")

def load_results(results_dir):
    """Load JSON result files from the results directory into a DataFrame."""
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files.")
    all_data = {}
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            print(f"Loaded {file} with {len(data)} keys.")
            # Optionally filter only keys that start with 'mean'
            data = {k: data[k] for k in data if k.startswith("mean")}
        label = os.path.basename(file)
        all_data[label] = data
    df = pd.DataFrame(all_data).T
    return df

def perform_analysis(df, image_filename):
    """Perform statistical analysis and generate visualizations from the DataFrame."""
    df.columns = df.columns.str.replace("mean_number_inliers_", "")
    df.columns = df.columns.str.replace("line_RANSAC_", "RANSAC-LP-")
    df.columns = df.columns.str.replace("standard_RANSAC_", "RANSAC-")

    ordered_columns = [
        "RANSAC-LP-957", "RANSAC-LP-747", "RANSAC-LP-558",
        "RANSAC-LP-388", "RANSAC-LP-239", "RANSAC-LP-109",
        "RANSAC-957", "RANSAC-747", "RANSAC-558",
        "RANSAC-388", "RANSAC-239", "RANSAC-109"
    ]
    df = df.loc[:, ordered_columns]
    df = df.div(df["RANSAC-109"], axis=0)
    
    print("Performing Friedman Test...")
    test_result = friedmanTest(df)
    print(test_result)
    
    print("Performing Friedman Post-hoc Test...")
    post_result = friedmanPost(df)
    print(post_result)
    
    adjusted_p_values = adjustShaffer(post_result)
    print("Adjusted p-values:")
    print(adjusted_p_values)
    
    # scmamp_results = friedman_shaffer_scmamp(df)
    # results_df = scmamp_results["adjusted_pvalues"]
    results_df = pd.DataFrame(adjusted_p_values)
    adjusted_p_values_df = results_df.copy()
    for i in range(len(adjusted_p_values_df)):
        adjusted_p_values_df.iloc[i, i] = ""
    
    means = df.mean()
    results_df_heatmap_data = to_heatmap_custom(results_df, means)
    print("Heatmap data:")
    print(results_df_heatmap_data)
    
    to_image_custom(results_df, results_df_heatmap_data, image_filename, df_p_values=adjusted_p_values_df)
    
    percentage_df = (df - 1) * 100
    mean_percentage = percentage_df.mean(axis=0)
    std_percentage = percentage_df.std(axis=0)
    
    ranks_df = df.rank(axis=1, method='average', ascending=False)
    mean_ranks = ranks_df.mean()
    std_ranks = ranks_df.std()
    
    summary = {
        "friedman_test": test_result,
        "friedman_post": post_result,
        "adjusted_p_values": adjusted_p_values,
        "mean_percentage": mean_percentage.to_dict(),
        "std_percentage": std_percentage.to_dict(),
        "mean_ranks": mean_ranks.to_dict(),
        "std_ranks": std_ranks.to_dict()
    }
    return summary

def load_time_data(results_dir):
    """
    Load execution times from JSON files in results_dir.
    Looks for keys starting with "standard_RANSAC_" and "line_RANSAC_".
    
    Returns:
        pd.DataFrame with columns: ['algorithm', 'iterations', 'time']
    """
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    time_records = []
    
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
        for key, results in data.items():
            # Process Standard RANSAC times.
            if key.startswith("standard_RANSAC_") and isinstance(results, list):
                # Extract iteration count from key.
                iteration = int(key.split("_")[-1])
                for rep in results:
                    if "time" in rep:
                        time_records.append({
                            "algorithm": "Standard RANSAC",
                            "iterations": iteration,
                            "time": rep["time"]
                        })
            # Process RANSACLP times.
            elif key.startswith("line_RANSAC_") and isinstance(results, list):
                iteration = int(key.split("_")[-1])
                for rep in results:
                    if "time" in rep:
                        time_records.append({
                            "algorithm": "RANSACLP",
                            "iterations": iteration,
                            "time": rep["time"]
                        })
    return pd.DataFrame(time_records)

def plot_time_boxplots(df_times, boxplot_filename="time_boxplots.png"):
    """
    Plot boxplots for execution time distributions.
    
    Parameters:
        df_times (pd.DataFrame): DataFrame with columns 'algorithm', 'iterations', and 'time'
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="iterations", y="time", hue="algorithm", data=df_times, palette="Set2")
    plt.title("Execution Time Distribution per Iteration Level and Algorithm")
    plt.xlabel("Iteration Level")
    plt.ylabel("Execution Time (seconds)")
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(boxplot_filename, dpi=300)
    # plt.show()

def load_inlier_data(results_dir):
    """
    Load number of inliers from JSON files in results_dir.
    Looks for keys starting with "standard_RANSAC_" and "line_RANSAC_".
    
    Returns:
        pd.DataFrame with columns: ['algorithm', 'iterations', 'inliers']
    """
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    inlier_records = []
    
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
        for key, results in data.items():
            # Process Standard RANSAC inliers.
            if key.startswith("standard_RANSAC_") and isinstance(results, list):
                iteration = int(key.split("_")[-1])
                for rep in results:
                    if "number_inliers" in rep:
                        inlier_records.append({
                            "algorithm": "Standard RANSAC",
                            "iterations": iteration,
                            "inliers": rep["number_inliers"]
                        })
            # Process RANSACLP inliers.
            elif key.startswith("line_RANSAC_") and isinstance(results, list):
                iteration = int(key.split("_")[-1])
                for rep in results:
                    if "number_inliers" in rep:
                        inlier_records.append({
                            "algorithm": "RANSACLP",
                            "iterations": iteration,
                            "inliers": rep["number_inliers"]
                        })
    return pd.DataFrame(inlier_records)

def plot_inlier_boxplots(df_inliers, boxplot_filename="inliers_boxplots.png"):
    """
    Plot boxplots for inlier counts distributions.
    
    Parameters:
        df_inliers (pd.DataFrame): DataFrame with columns 'algorithm', 'iterations', and 'inliers'
        boxplot_filename (str): Filename to save the plot.
    """
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x="iterations", y="inliers", hue="algorithm", data=df_inliers, palette="Set2")
    sns.stripplot(x="iterations", y="inliers", hue="algorithm", data=df_inliers,
                  dodge=True, jitter=True, color="black", size=3, ax=ax)
    plt.title("Distribution of Inliers per Iteration Level and Algorithm")
    plt.xlabel("Iteration Level")
    plt.ylabel("Number of Inliers")
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(boxplot_filename, dpi=300)
    # plt.show()

def load_time_data(results_dir):
    """
    Load execution times from JSON files in results_dir.
    Looks for keys starting with "standard_RANSAC_" and "line_RANSAC_", and now also
    the planar patches detection time.
    
    Returns:
        pd.DataFrame with columns: ['algorithm', 'iterations', 'time']
    """
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    time_records = []
    
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
        for key, results in data.items():
            # Standard RANSAC times.
            if key.startswith("standard_RANSAC_") and isinstance(results, list):
                iteration = int(key.split("_")[-1])
                for rep in results:
                    if "time" in rep:
                        time_records.append({
                            "algorithm": "Standard RANSAC",
                            "iterations": iteration,
                            "time": rep["time"]
                        })
            # RANSACLP times.
            elif key.startswith("line_RANSAC_") and isinstance(results, list):
                iteration = int(key.split("_")[-1])
                for rep in results:
                    if "time" in rep:
                        time_records.append({
                            "algorithm": "RANSACLP",
                            "iterations": iteration,
                            "time": rep["time"]
                        })
            # Planar patches detection time.
            elif key == "planar_patches" and isinstance(results, dict):
                # We assign a dummy iteration level, e.g. 0.
                time_records.append({
                    "algorithm": "Planar Patches",
                    "iterations": 0,
                    "time": results.get("detection_time", None)
                })
    return pd.DataFrame(time_records)

def load_inlier_data(results_dir):
    """
    Load number of inliers from JSON files in results_dir.
    Looks for keys starting with "standard_RANSAC_" and "line_RANSAC_", and also extracts the best
    planar patches inliers.
    
    Returns:
        pd.DataFrame with columns: ['algorithm', 'iterations', 'inliers']
    """
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    inlier_records = []
    
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
        for key, results in data.items():
            # Standard RANSAC inliers.
            if key.startswith("standard_RANSAC_") and isinstance(results, list):
                iteration = int(key.split("_")[-1])
                for rep in results:
                    if "number_inliers" in rep:
                        inlier_records.append({
                            "algorithm": "Standard RANSAC",
                            "iterations": iteration,
                            "inliers": rep["number_inliers"]
                        })
            # RANSACLP inliers.
            elif key.startswith("line_RANSAC_") and isinstance(results, list):
                iteration = int(key.split("_")[-1])
                for rep in results:
                    if "number_inliers" in rep:
                        inlier_records.append({
                            "algorithm": "RANSACLP",
                            "iterations": iteration,
                            "inliers": rep["number_inliers"]
                        })
            # Best planar patches inliers.
            elif key == "planar_patches" and isinstance(results, dict):
                best_patch = results.get("best_patch", None)
                if best_patch is not None and "number_inliers" in best_patch:
                    inlier_records.append({
                        "algorithm": "Planar Patches",
                        "iterations": 0,
                        "inliers": best_patch["number_inliers"]
                    })
    return pd.DataFrame(inlier_records)


def main():
    o3d.utility.random.seed(42)
    # Configuration parameters.
    # repetitions = 10
    # iterations_list = [100, 200, 300, 400, 500, 600]
    repetitions = 10
    iterations_list = [100, 200, 300, 400, 500, 600]
    threshold = 0.02
    percentage_chosen_lines = 0.2
    percentage_chosen_planes = 0.05
    seed = 42
    results_dir = "/tmp/results_s3dis_json"
    image_filename = "/tmp/results_s3dis_json/shaffer_s3dis.png"
    boxplot_filename = "/tmp/results_s3dis_json/time_boxplots_s3dis.png"
    
    # Optionally, set source_dir to a directory path to search for files.
    # If source_dir is None, the code will use Open3D's built-in datasets.
    # source_dir = None
    # Uncomment the following line to use a directory instead:
    # source_dir = "/tmp/Downloads/PLY"
    source_dir = "/tmp/S3DIS/Stanford3dDataset_v1.2"
    # source_dir = "/tmp/Stanford3dDataset_v1.2"
    
    run_experiments(results_dir, repetitions, iterations_list, threshold,
                    percentage_chosen_lines, percentage_chosen_planes, seed, source_dir)
    
    df = load_results(results_dir)
    print("Loaded DataFrame:")
    print(df)
    
    summary = perform_analysis(df, image_filename)
    print("Analysis Summary:")
    print(summary)

    # Load time data and plot boxplots.
    df_times = load_time_data(results_dir)
    if df_times.empty:
        print("No execution time data found.")
    else:
        print("Time data:")
        print(df_times.head())
        plot_time_boxplots(df_times, boxplot_filename)

    df_inliers = load_inlier_data(results_dir)
    if df_inliers.empty:
        print("No inlier data found.")
    else:
        print("Inlier data:")
        print(df_inliers.head())
        plot_inlier_boxplots(df_inliers, "/tmp/results_s3dis_json/inliers_boxplots_s3dis.png")
        print("Inlier boxplot saved as inliers_boxplots.png")

if __name__ == "__main__":
    main()
