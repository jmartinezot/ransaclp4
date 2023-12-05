import ransaclp
import open3d as o3d
from lanverso_industrial_dataset_helper.lanverso_industrial_dataset_helper import IndustrialDataset as industrial_dataset


ransaclp_iterations = 200
threshold = 0.001
seed = 42
percentage_chosen_lines = 0.2
percentage_chosen_planes = 0.05

dataset = industrial_dataset()
pcd = o3d.io.read_point_cloud(dataset.paths[0])

plane_model, ransaclp_inliers = ransaclp.segment_plane(pcd, distance_threshold=threshold, num_iterations=ransaclp_iterations,
                                                percentage_chosen_lines = percentage_chosen_lines, use_cuda = True,
                                                percentage_chosen_planes = percentage_chosen_planes, seed = seed)
number_inliers = len(ransaclp_inliers)
ransaclp_inlier_cloud = pcd.select_by_index(ransaclp_inliers)
ransaclp_inlier_cloud.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd, ransaclp_inlier_cloud], window_name="RANSACLP Inliers:  " + str(number_inliers))


plane_model, o3d_inliers = pcd.segment_plane(distance_threshold=threshold,
                                         ransac_n=3,
                                         num_iterations=ransaclp_iterations,
                                         probability=1)
o3d_inlier_cloud = pcd.select_by_index(o3d_inliers)
o3d_inlier_cloud.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd, o3d_inlier_cloud], window_name="Open3D Inliers:  " + str(number_inliers))

common_inliers = list(set(ransaclp_inliers).intersection(o3d_inliers))
common_inlier_cloud = pcd.select_by_index(common_inliers)
common_inlier_cloud.paint_uniform_color([1, 0, 0])
only_ransaclp_inliers = list(set(ransaclp_inliers) - set(o3d_inliers))
ransaclp_inlier_cloud = pcd.select_by_index(only_ransaclp_inliers)
ransaclp_inlier_cloud.paint_uniform_color([0, 1, 0])
only_o3d_inliers = list(set(o3d_inliers) - set(ransaclp_inliers))
o3d_inlier_cloud = pcd.select_by_index(only_o3d_inliers)
o3d_inlier_cloud.paint_uniform_color([0, 0, 1])
o3d.visualization.draw_geometries([pcd, common_inlier_cloud, ransaclp_inlier_cloud, o3d_inlier_cloud], window_name="R: common; G: RANSACLP; B: Open3D")
