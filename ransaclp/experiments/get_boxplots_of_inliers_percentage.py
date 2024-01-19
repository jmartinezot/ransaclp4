import pickle as pkl
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pkl.load(file)

def collect_means_from_pickles(pkl_files):
    mean_data = {}
    percentage_data = {}
    for pkl_file in pkl_files:
        data = load_pickle(pkl_file)
        total_points = data['number_pcd_points']
        for key in data.keys():
            if key.startswith('mean_number_inliers'):
                if 'line_RANSAC' in key:
                    transformed_key = key.replace('mean_number_inliers_line_RANSAC', 'RANSACLP')
                elif 'standard_RANSAC' in key:
                    transformed_key = key.replace('mean_number_inliers_standard_RANSAC', 'RANSAC')
                else:
                    continue

                if transformed_key not in mean_data:
                    mean_data[transformed_key] = []
                mean_data[transformed_key].append(data[key])

                if transformed_key not in percentage_data:
                    percentage_data[transformed_key] = []
                percentage_data[transformed_key].append((data[key] / total_points) * 100)

    return percentage_data

def plot_boxplots_with_annotations_seaborn(percentage_data, save_path=None):
    # Sorting the data in descending order based on median values
    sorted_data = sorted(percentage_data.items(), key=lambda x: np.median(x[1]), reverse=True)

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

    plt.ylabel('Percentage of Inliers')
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

pickes_path = "/home/scpmaotj/Github/ransaclp/ransaclp/results_experiments_ransaclp/S3DIS"
# pickes_path = "/home/scpmaotj/Github/ransaclp/ransaclp/results_experiments_ransaclp/Open3D"
# pickes_path = "/home/scpmaotj/Github/ransaclp/ransaclp/results_experiments_ransaclp/Tecnalia"
pkl_files = glob.glob(pickes_path + "/**/*.pkl", recursive=True)
percentage_data = collect_means_from_pickles(pkl_files)
plot_boxplots_with_annotations_seaborn(percentage_data, save_path='boxplots_percentaje_s3dis.png')

latex_table = create_latex_table(percentage_data)
print(latex_table)