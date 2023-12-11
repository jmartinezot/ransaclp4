The file "results_baseline_S3DIS.pkl" is the results of open3d with 100,000 iterations over the S3DIS database. The code for generating is in the docstring of "get_baseline_S3DIS" in "ransaclpexperiments.py".
The pkl files in Open3D are the results after applying the algorithm over the database of Open3D. The file for generating this data is "experiment_all_files_Open3D_RANSAC_lines_fitting_plane.py"
The pkl files in S3DIS are the results after applying the algorithm over the database of S3DIS. The file for generating this data is "experiment_all_files_S3DIS_RANSAC_lines_fitting_plane.py"
To perform the Friedman test and Shaffer posthoc, run "friedman_and_shaffer_any_directory.py".
The image is saved in the current directory.
If you run the script inside iPython, just check mean_ranks and std_ranks for building the table with mean ranks. The percentage data is in mean_percentage_df and std_percentage_df.
For the Friedman statistics, check test_results.
