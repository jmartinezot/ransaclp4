
import pickle
import pandas as pd
import glob
from . import stats

results_path = "/home/scpmaotj/Github/ransaclp/ransaclp/results_experiments_ransaclp/Tecnalia/"
image_filename = "shaffer_tecnalia.png"
# results_path = "/home/scpmaotj/Github/mrdja/results_experiments_ransaclp/S3DIS" 
# image_filename = "shaffer_s3dis.png"
pkl_files = glob.glob(results_path + "/**/*.pkl", recursive=True)

# Initialize an empty dictionary to hold all the data
all_data = {}
  
for file_path in pkl_files:
    # Load data from the current file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        # data is a dict. Remove all the keys that are not mean*
        data = {key: data[key] for key in data.keys() if key.startswith("mean")}
    print(data)
    # retain only the last two sub directories and the filename of file_path
    file_path = "/".join(file_path.split("/")[-3:])
    all_data[file_path] = data

# Convert the dictionary of dictionaries into a DataFrame

df = pd.DataFrame(all_data).T

# retain only the 20 first rows
# df = df.iloc[:10]

# remove "mean\_number\_inliers\_line\_" from the start of the column names
df.columns = df.columns.str.replace("mean_number_inliers_", "")
df.columns = df.columns.str.replace("line_RANSAC_", "RANSAC-LP-")
df.columns = df.columns.str.replace("standard_RANSAC_", "RANSAC-")

ordered_columns = ["RANSAC-LP-957", "RANSAC-LP-747", "RANSAC-LP-558", "RANSAC-LP-388", "RANSAC-LP-239", "RANSAC-LP-109", 
                "RANSAC-957", "RANSAC-747", "RANSAC-558", "RANSAC-388", "RANSAC-239", "RANSAC-109"]
df = df.loc[:,ordered_columns]

# divide all the columns by the column "RANSAC-109"
# df = df.div(df["RANSAC-109"], axis=0)
df = df.div(df["RANSAC-109"], axis=0)

# Hardcoded sample data
'''
data = np.array([
    [75.8, 79.4, 76.4, 77.9, 78.5, 76.2, 75.1, 78.4, 75.6, 70.3],
    [8.7, 6.6, 9.8, 7.7, 7.2, 8.4, 5.9, 9.3, 8.1, 60.5],
    [9.0, 5.4, 7.8, 6.8, 8.0, 7.5, 6.3, 9.1, 5.2, 70.6]
]).T
'''

# Convert the numpy array into a DataFrame
# df = pd.DataFrame(data, columns=["A", "B", "C"])

# df = df.reset_index(drop=True)
# df.to_csv('temp_data.csv', index=False)

only_test_result = stats.friedmanTest(df)
print(only_test_result)
only_post_result = stats.friedmanPost(df)
print(only_post_result)
adjusted_p_values = stats.adjustShaffer(only_post_result)
print(adjusted_p_values)

# call the stats function to perform the Friedman test con Shaffer's post-hoc test
test_results = stats.friedman_shaffer_scmamp(df)
results_df = test_results["adjusted_pvalues"]
adjusted_p_values_df = results_df.copy() # make a copy of the results_df DataFrame
# Replace NaN values in the diagonal with empty strings
for i in range(len(adjusted_p_values_df)):
    adjusted_p_values_df.iloc[i, i] = ""

# Compute mean values for each algorithm
means = df.mean()
results_df_heatmap_data = stats.to_heatmap_custom(results_df, means)
print("results_df_heatmap_data")
print(results_df_heatmap_data)
stats.to_image_custom(results_df, results_df_heatmap_data, image_filename, df_p_values = adjusted_p_values_df)

# substract 1 from all the columns and multiply by 100
percentage_df = (df - 1) * 100
# get the mean of all columns
mean_percentage_df = percentage_df.mean(axis=0)
# get the standard deviation of all columns
std_percentage_df = percentage_df.std(axis=0)

ranks_df = df.rank(axis=1, method='average', ascending=False)
mean_ranks = ranks_df.mean()
std_ranks = ranks_df.std()
