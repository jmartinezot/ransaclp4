import ransaclp
import open3d as o3d

office_dataset = o3d.data.OfficePointClouds()
office_filename = office_dataset.paths[0]

ransaclp_iterations = 200
threshold = 0.02
seed = 42
percentage_chosen_lines = 0.2
percentage_chosen_planes = 0.05

pcd = o3d.io.read_point_cloud(office_filename)
plane_model, inliers = ransaclp.segment_plane(pcd, distance_threshold=threshold, num_iterations=ransaclp_iterations, 
                                                percentage_chosen_lines = percentage_chosen_lines,
                                                percentage_chosen_planes = percentage_chosen_planes, seed = seed)
number_inliers = len(inliers)
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd, inlier_cloud], window_name="RANSACLP Inliers:  " + str(number_inliers))

