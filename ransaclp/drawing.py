'''
This module contains functions for drawing 3D objects using matplotlib.
'''
import numpy as np
import open3d as o3d
from . import geometry as geom
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

def draw_plane_as_lines_open3d(A, B, C, D, size=10, line_color=[1, 0, 0], grid_density=5, external_point=None):
    vertices = []
    lines = []
    plane = np.array([A, B, C, D])
    
    if external_point is not None:
        center = geom.get_point_of_plane_closest_to_given_point(plane, external_point)
        center_x, center_y, center_z = center[0], center[1], center[2]
    else:
        center_x, center_y, center_z = 0, 0, 0
    
    # Add the corners of the grid
    for i in np.linspace(-size, size, grid_density):
        for j in np.linspace(-size, size, grid_density):
            x = center_x + i
            y = center_y + j
            z = -(D + A * x + B * y) / C
            vertices.append([x, y, z])

    n = len(vertices)
    vertices = np.array(vertices)
    
    # Add the lines for the grid (horizontal and vertical)
    for i in range(grid_density):
        for j in range(grid_density - 1):
            # horizontal lines
            lines.append([i * grid_density + j, i * grid_density + j + 1])
            # vertical lines
            lines.append([j * grid_density + i, (j + 1) * grid_density + i])

    lines = np.array(lines)
    
    # Create the LineSet
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    
    # Set the color for each line
    colors = [line_color for _ in range(len(lines))]
    lineset.colors = o3d.utility.Vector3dVector(colors)

    return lineset

