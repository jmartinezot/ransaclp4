import os
import json
import glob
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
from matplotlib.ticker import StrMethodFormatter
from numba import cuda

sns.set_palette("colorblind")

def annotate_boxplot_medians(ax, df, x_col, y_col, hue_col):
    """
    Annotate each box in a Seaborn boxplot with its median value.
    
    Parameters:
      ax      : The Axes object on which the boxplot is drawn.
      df      : The DataFrame used to plot the boxplot.
      x_col   : The name of the categorical variable (x-axis).
      y_col   : The name of the numeric variable (y-axis).
      hue_col : The name of the grouping variable (hue).
    """
    # Group the data to compute medians for each combination.
    medians = df.groupby([x_col, hue_col])[y_col].median().reset_index()
    
    # Determine the x-coordinate positions for each box.
    # Seaborn's boxplot returns a list of patches for the boxes.
    # We'll iterate over the patches and annotate based on their position.
    # This approach works if you know that the patches are in the same order as the grouped medians.
    patches = ax.artists
    num_boxes = len(patches)
    
    # For each patch, extract its x-position and width, then place the median text above.
    for i, patch in enumerate(patches):
        # Get the x-position and width of the box
        x = patch.get_x()
        width = patch.get_width()
        # Compute the center x-position for the annotation
        center_x = x + width / 2.0
        
        # Retrieve the median value for this box.
        # This assumes that the order of patches corresponds to the order in 'medians'.
        median_val = medians.iloc[i][y_col]
        
        # Add text annotation above the box with two decimal places.
        ax.text(center_x, median_val, f"{median_val:.2f}",
                ha="center", va="bottom", fontsize=10, color="black")


def make_json_serializable(obj):
    """Recursively convert non-JSON-serializable objects (e.g., NumPy arrays) to JSON-friendly types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    return obj

def load_results(results_dir):
    """Load JSON result files from the results directory into a DataFrame."""
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files.")
    all_data = {}
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
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
    Returns a DataFrame with columns: ['algorithm', 'iterations', 'time']
    """
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    time_records = []
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
        for key, results in data.items():
            if key.startswith("standard_RANSAC_") and isinstance(results, list):
                iteration = int(key.split("_")[-1])
                for rep in results:
                    if "time" in rep:
                        time_records.append({
                            "algorithm": "Standard RANSAC",
                            "iterations": iteration,
                            "time": rep["time"]
                        })
            elif key.startswith("line_RANSAC_") and isinstance(results, list):
                iteration = int(key.split("_")[-1])
                for rep in results:
                    if "time" in rep:
                        time_records.append({
                            "algorithm": "RANSAC-LP4",
                            "iterations": iteration,
                            "time": rep["time"]
                        })
            elif key == "planar_patches" and isinstance(results, dict):
                time_records.append({
                    "algorithm": "Planar Patches",
                    "iterations": 0,
                    "time": results.get("detection_time", None)
                })
    return pd.DataFrame(time_records)

def plot_time_boxplots(df_times, boxplot_filename="time_boxplots.png"):
    """
    Plot boxplots for execution time distributions.
    Filters out entries for Planar Patches (which have iterations==0).
    """
    df_plot = df_times[df_times['iterations'] != 0]
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="iterations", y="time", hue="algorithm", data=df_plot, 
                palette="colorblind", showfliers=False)
    plt.title("Execution Time Distribution per Iteration Level and Algorithm")
    plt.xlabel("Iteration Level")
    plt.ylabel("Execution Time (seconds)")
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(boxplot_filename, dpi=300)
    plt.close()

'''
def plot_time_boxplots_dual_scale(df_times, boxplot_filename="time_boxplots_dual.png"):
    """
    Plot boxplots for execution time distributions with two different scales:
      - Left subplot: Standard RANSAC and RANSAC-LP4 (iterations != 0) with outliers removed.
      - Right subplot: Boxplot for Planar Patches (iterations == 0) with its own y-axis scale and a thinner box.
    A vertical dashed line is added between the subplots.
    Each box is annotated with its median value (formatted to two decimals) above the box.
    """
    from matplotlib.ticker import StrMethodFormatter

    # Split data
    df_nonplanar = df_times[df_times['iterations'] != 0]
    df_planar = df_times[df_times['iterations'] == 0]
    
    # Create dual subplot with different width ratios
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False,
                             gridspec_kw={'width_ratios': [3, 1]})
    
    # Left subplot: Standard RANSAC and RANSAC-LP4
    ax0 = axes[0]
    sns.boxplot(x="iterations", y="time", hue="algorithm", data=df_nonplanar,
                palette="colorblind", ax=ax0, showfliers=False)
    ax0.set_title("Standard RANSAC and RANSAC-LP4")
    ax0.set_xlabel("Iteration Level")
    ax0.set_ylabel("Execution Time (seconds)")
    ax0.legend(title="Algorithm")
    
    # Annotate medians on left subplot
    # Compute medians grouped by iterations and algorithm; assume ordering matches boxplot artists.
    medians0 = df_nonplanar.groupby(["iterations", "algorithm"])["time"].median().reset_index()
    for i, artist in enumerate(ax0.artists):
        x = artist.get_x()
        width = artist.get_width()
        center_x = x + width / 2.0
        median_val = medians0.iloc[i]["time"]
        ax0.text(center_x, median_val, f"{median_val:.2f}",
                 ha="center", va="bottom", fontsize=9, color="black")
    
    # Right subplot: Planar Patches (single category)
    ax1 = axes[1]
    sns.boxplot(x=["Planar Patches"] * len(df_planar), y="time", data=df_planar,
                width=0.2, palette="colorblind", ax=ax1, showfliers=False)
    ax1.set_title("Planar Patches")
    ax1.set_xlabel("")
    ax1.set_ylabel("Execution Time (seconds)")
    ax1.set_xticklabels([])
    median_planar = df_planar["time"].median()
    if ax1.artists:
        artist = ax1.artists[0]
        x = artist.get_x()
        width = artist.get_width()
        center_x = x + width / 2.0
        ax1.text(center_x, median_planar, f"{median_planar:.2f}",
                 ha="center", va="bottom", fontsize=9, color="black")
    
    # Add vertical dashed line between subplots
    pos1 = ax1.get_position()
    # Place the line slightly left of the left edge of the right subplot
    x_line = pos1.x0 - 0.01  
    pos0 = ax0.get_position()
    fig.lines.append(plt.Line2D([x_line, x_line], [pos0.y0, pos0.y1],
                                  color="black", linestyle="--", transform=fig.transFigure))
    
    plt.tight_layout()
    plt.savefig(boxplot_filename, dpi=300)
    plt.close()
'''

def annotate_boxplot_medians(ax, df, groupby_cols, value_col):
    """
    Annotate each box in a grouped Seaborn boxplot with its median value.
    
    Parameters:
      ax         : The Axes object on which the boxplot is drawn.
      df         : The DataFrame used to plot the boxplot.
      groupby_cols: List of columns to group by (e.g., [hue_col, x_col]).
      value_col  : The column name containing the numeric values.
    """
    # Group and sort the data by the iteration level (assumed second element) and algorithm (first element)
    grouped = df.groupby(groupby_cols)[value_col].median().reset_index()
    grouped = grouped.sort_values(by=[groupby_cols[1], groupby_cols[0]]).reset_index(drop=True)
    
    # Force drawing of the canvas so that positions are updated.
    plt.draw()
    
    # Iterate over the box artists in the axis.
    for i, patch in enumerate(ax.artists):
        # Get x position and width of the box
        x = patch.get_x()
        width = patch.get_width()
        center_x = x + width / 2.0
        # Retrieve the corresponding median from our grouped DataFrame
        try:
            median_val = grouped.iloc[i][value_col]
        except IndexError:
            continue  # In case there is a mismatch in the number of boxes
        ax.text(center_x, median_val, f"{median_val:.2f}",
                ha="center", va="bottom", fontsize=9, color="black")

def plot_time_boxplots_dual_scale(df_times, boxplot_filename="time_boxplots_dual.png"):
    """
    Plot boxplots for execution time distributions with two different scales:
      - Left subplot: Standard RANSAC and RANSAC-LP4 (iterations != 0) with outliers removed.
      - Right subplot: Boxplot for Planar Patches (iterations == 0) with its own y-axis scale and a thinner box.
    Each box is annotated with its median value (formatted to two decimals) above the box.
    """
    from matplotlib.ticker import StrMethodFormatter

    # Split data
    df_nonplanar = df_times[df_times['iterations'] != 0]
    df_planar = df_times[df_times['iterations'] == 0]
    
    # Create dual subplot with different width ratios
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False,
                             gridspec_kw={'width_ratios': [3, 1]})
    
    # Left subplot: Standard RANSAC and RANSAC-LP4.
    ax0 = axes[0]
    sns.boxplot(x="iterations", y="time", hue="algorithm", data=df_nonplanar,
                palette="colorblind", ax=ax0, showfliers=False)
    ax0.set_title("Standard RANSAC and RANSAC-LP4")
    ax0.set_xlabel("Iteration Level")
    ax0.set_ylabel("Execution Time (seconds)")
    ax0.legend(title="Algorithm")
    
    # Right subplot: Planar Patches (single category, thin box).
    ax1 = axes[1]
    sns.boxplot(x=["Planar Patches"] * len(df_planar), y="time", data=df_planar,
                width=0.2, palette="colorblind", ax=ax1, showfliers=False)
    ax1.set_title("Planar Patches")
    ax1.set_xlabel("")
    ax1.set_ylabel("Execution Time (seconds)")
    ax1.set_xticklabels([])
    median_planar = df_planar["time"].median()
    if ax1.artists:
        patch = ax1.artists[0]
        x = patch.get_x()
        width = patch.get_width()
        center_x = x + width / 2.0
        ax1.text(center_x, median_planar, f"{median_planar:.2f}",
                 ha="center", va="bottom", fontsize=9, color="black")
    
    # Removed vertical dashed line code.
    
    plt.tight_layout()
    plt.savefig(boxplot_filename, format="eps", dpi=300)
    plt.close()



def load_inlier_data(results_dir):
    """
    Load number of inliers from JSON files in results_dir.
    Returns a DataFrame with columns: ['algorithm', 'iterations', 'inliers']
    """
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    inlier_records = []
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
        for key, results in data.items():
            if key.startswith("standard_RANSAC_") and isinstance(results, list):
                iteration = int(key.split("_")[-1])
                for rep in results:
                    if "number_inliers" in rep:
                        inlier_records.append({
                            "algorithm": "Standard RANSAC",
                            "iterations": iteration,
                            "inliers": rep["number_inliers"]
                        })
            elif key.startswith("line_RANSAC_") and isinstance(results, list):
                iteration = int(key.split("_")[-1])
                for rep in results:
                    if "number_inliers" in rep:
                        inlier_records.append({
                            "algorithm": "RANSAC-LP4",
                            "iterations": iteration,
                            "inliers": rep["number_inliers"]
                        })
            elif key == "planar_patches" and isinstance(results, dict):
                best_patch = results.get("best_patch", None)
                if best_patch is not None and "number_inliers" in best_patch:
                    inlier_records.append({
                        "algorithm": "Planar Patches",
                        "iterations": 0,
                        "inliers": best_patch["number_inliers"]
                    })
    return pd.DataFrame(inlier_records)

def plot_inlier_boxplots(df_inliers, boxplot_filename="inliers_boxplots.png"):
    """
    Plot boxplots for inlier count distributions.
    Uses a formatter to display inlier counts with commas (e.g., 100,000).
    Filters out entries for Planar Patches (iterations == 0).
    """
    from matplotlib.ticker import StrMethodFormatter
    df_plot = df_inliers[df_inliers['iterations'] != 0]
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x="iterations", y="inliers", hue="algorithm", data=df_plot, 
                     palette="colorblind", showfliers=False)
    plt.title("Distribution of Inliers per Iteration Level and Algorithm")
    plt.xlabel("Iteration Level")
    plt.ylabel("Number of Inliers")
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(boxplot_filename, dpi=300)
    plt.close()

def plot_inlier_boxplots_side_by_side(df_inliers, boxplot_filename="inliers_boxplots_side_by_side.png"):
    """
    Plot side-by-side boxplots for inlier count distributions:
      - Left subplot: Boxplots for Standard RANSAC and RANSAC-LP4 (grouped by iteration level, outliers removed).
      - Right subplot: Boxplot for Planar Patches (single category) with a thin box.
    Uses a formatter to display inlier counts with commas.
    """
    from matplotlib.ticker import StrMethodFormatter
    df_others = df_inliers[df_inliers['algorithm'] != 'Planar Patches'].copy()
    df_planar = df_inliers[df_inliers['algorithm'] == 'Planar Patches'].copy()
    df_planar["Category"] = "Planar Patches"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    
    sns.boxplot(x="iterations", y="inliers", hue="algorithm", data=df_others, 
                palette="colorblind", ax=axes[0], showfliers=False)
    axes[0].set_title("Standard RANSAC and RANSAC-LP4")
    axes[0].set_xlabel("Iteration Level")
    axes[0].set_ylabel("Number of Inliers")
    axes[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axes[0].legend(title="Algorithm")
    
    sns.boxplot(x="Category", y="inliers", data=df_planar, 
                palette="colorblind", ax=axes[1], width=0.2, showfliers=False)
    axes[1].set_title("Planar Patches")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Number of Inliers")
    axes[1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(boxplot_filename, dpi=300)
    plt.close()

def main():
    results_dir = "/tmp/results_tecnalia_json"
    image_filename = os.path.join(results_dir, "shaffer_tecnalia.png")
    boxplot_time_filename = os.path.join(results_dir, "time_boxplots_tecnalia.png")
    boxplot_time_dual_filename = os.path.join(results_dir, "time_boxplots_dual_tecnalia.eps")
    boxplot_inlier_filename = os.path.join(results_dir, "inliers_boxplots_tecnalia.png")
    boxplot_inlier_side_filename = os.path.join(results_dir, "inliers_boxplots_side_by_side_tecnalia.png")
    
    # Load and analyze results.
    df = load_results(results_dir)
    print("Loaded DataFrame:")
    print(df.head())
    
    summary = perform_analysis(df, image_filename)
    print("Analysis Summary:")
    print(summary)
    
    # Plot time boxplots.
    df_times = load_time_data(results_dir)
    if df_times.empty:
        print("No execution time data found.")
    else:
        print("Time data:")
        print(df_times.head())
        plot_time_boxplots(df_times, boxplot_time_filename)
        plot_time_boxplots_dual_scale(df_times, boxplot_time_dual_filename)
    
    # Plot inlier boxplots.
    df_inliers = load_inlier_data(results_dir)
    if df_inliers.empty:
        print("No inlier data found.")
    else:
        print("Inlier data:")
        print(df_inliers.head())
        plot_inlier_boxplots(df_inliers, boxplot_inlier_filename)
        plot_inlier_boxplots_side_by_side(df_inliers, boxplot_inlier_side_filename)
        print(f"Inlier boxplots saved as {boxplot_inlier_filename} and {boxplot_inlier_side_filename}")

if __name__ == "__main__":
    main()
