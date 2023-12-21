import pickle as pkl
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pkl.load(file)

def collect_ranks_from_pickles(pkl_files):
    rank_data = {}
    for pkl_file in pkl_files:
        data = load_pickle(pkl_file)

        # Create a dictionary to store the inliers count for each algorithm
        inliers_data = {}
        for key in data.keys():
            if key.startswith('mean_number_inliers'):
                # Transform the key to a more readable format
                if 'line_RANSAC' in key:
                    transformed_key = key.replace('mean_number_inliers_line_RANSAC', 'RANSACLP')
                elif 'standard_RANSAC' in key:
                    transformed_key = key.replace('mean_number_inliers_standard_RANSAC', 'RANSAC')
                else:
                    continue

                inliers_data[transformed_key] = data[key]

        # Sort the algorithms based on inliers count and get their ranks
        sorted_algorithms = sorted(inliers_data, key=inliers_data.get, reverse=True)
        for rank, algorithm in enumerate(sorted_algorithms, start=1):
            if algorithm not in rank_data:
                rank_data[algorithm] = []
            rank_data[algorithm].append(rank)

    return rank_data

def plot_boxplots_with_annotations_seaborn(percentage_data, save_path=None):
    # Sorting the data in descending order based on median values
    sorted_data = sorted(percentage_data.items(), key=lambda x: np.median(x[1]), reverse=False)

    # Extracting sorted labels and values
    sorted_labels = [label for label, _ in sorted_data]
    sorted_values = [values for _, values in sorted_data]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the boxplot using Seaborn
    sns.boxplot(data=sorted_values, ax=ax)

    # Remove default x-tick labels
    ax.set_xticklabels([])

    # Format custom x-tick labels and highlight labels containing "RANSACLP" in bold
    for i, label in enumerate(sorted_labels):
        if "RANSACLP" in label:
            ax.text(i, ax.get_ylim()[0], label, rotation=45, ha='right', va='top', fontsize=10, fontweight='bold')
        else:
            ax.text(i, ax.get_ylim()[0], label, rotation=45, ha='right', va='top', fontsize=10)

    # Calculate and annotate median, Q1, Q3 for each set of sorted data
    for i, values in enumerate(sorted_values):
        median = np.median(values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)

        # Adjust text placement to be just below the lines
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02  # Small offset for text
        ax.text(i, median - offset, f'{median:.2f}', verticalalignment='top', horizontalalignment='center', color='black', fontsize=10)
        ax.text(i, q3 - offset, f'{q3:.2f}', verticalalignment='top', horizontalalignment='center', color='black', fontsize=10)
        ax.text(i, q1 - offset, f'{q1:.2f}', verticalalignment='top', horizontalalignment='center', color='black', fontsize=10)

    # plt.ylabel('Percentage of Inliers')
    # plt.ylabel('Percentage of Improvement over RANSAC_109')
    plt.ylabel('Rank')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=300)
    plt.show()

def create_latex_table(percentage_data):
    rows = []
    for algorithm, data in percentage_data.items():
        mean_val = np.mean(data)
        std_dev = np.std(data)
        rows.append([algorithm, mean_val, std_dev])
    
    df = pd.DataFrame(rows, columns=['Algorithm', 'Mean (%)', 'Standard Deviation (%)'])
    latex_table = df.to_latex(index=False, float_format="{:0.2f}".format)
    return latex_table

# pickes_path = "/home/scpmaotj/Github/ransaclp/ransaclp/results_experiments_ransaclp/S3DIS"
# pickes_path = "/home/scpmaotj/Github/ransaclp/ransaclp/results_experiments_ransaclp/Open3D"
pickes_path = "/home/scpmaotj/Github/ransaclp/ransaclp/results_experiments_ransaclp/Tecnalia"
pkl_files = glob.glob(pickes_path + "/**/*.pkl", recursive=True)
percentage_data = collect_ranks_from_pickles(pkl_files)
plot_boxplots_with_annotations_seaborn(percentage_data, save_path='boxplots_ranks_tecnalia.png')

latex_table = create_latex_table(percentage_data)
print(latex_table)