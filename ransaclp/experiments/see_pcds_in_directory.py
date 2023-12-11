# Use open3d to show all the pintclouds in a directory

import open3d as o3d
import os

directory = "/home/scpmaotj/Tecnalia_Lanverso_dataset/EvenTableTwoPartsRealsense/"
directory = "/home/scpmaotj/Tecnalia_Lanverso_dataset/EvenTableTwoPartsZivid/"
files = os.listdir(directory)

for file in files:
    if file.endswith(".pcd"):
        pcd = o3d.io.read_point_cloud(os.path.join(directory, file))
        o3d.visualization.draw_geometries([pcd], window_name=file)