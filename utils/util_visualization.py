import math
import numpy as np
import open3d as o3d
import open3d.visualization as vis
import os
import random

from run_nerf_helpers import *

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

def to_4x4(X): 
    # Convert 3x4 matrix to 4x4 one.
    shape = X.shape[:-2] + (4, 4,)
    
    X_4x4 = np.zeros(shape, dtype=np.float64)
    X_4x4[..., :3, :4] = X
    X_4x4[..., 3, 3] = 1.

    return X_4x4

def get_colormap(num_samples, cmap_name='gnuplot'):
    #x = np.linspace(0., 1., num_samples)
    x = np.linspace(0.2, 0.8, num_samples)
    #cmap = plt.get_cmap(cmap_name)
    #norm = mpl.colors.Normalize(vmin=0.4, vmax=0.6)
    scalarMap = cm.ScalarMappable(cmap=cmap_name)

    return scalarMap.to_rgba(x)[..., :3]

def voxel_grid_visualization(args, rgbs, depths, intrinsic, extrinsics, plenoxel, H, W, i_test, img_dir='', stride=8):
    
    if extrinsics.shape[-2:] == (3, 4):
        extrinsics = to_4x4(extrinsics)
    
    n_cams = extrinsics.shape[0]

    color_cams = get_colormap(n_cams)

    vis = o3d.visualization.Visualizer()
    
    vis.create_window(visible=False)
    # draw camera.
    for i_cam, extrinsic_i in enumerate(extrinsics):
        # draw camera pose.
        cam_obj_i = create_camera_lineset(intrinsic, extrinsic_i, H, W, color_cams[i_cam])
        
        vis.add_geometry(cam_obj_i)
    
    # draw voxel boundary.
    center = plenoxel.center.tolist()
    radius = plenoxel.radius.tolist()

    #color_lines = [255,255,255]
    lines = o3d.geometry.LineSet()
    
    lines.points = o3d.utility.Vector3dVector([[center[0]-radius[0],center[1]-radius[1],center[2]-radius[2]],
                                               [center[0]-radius[0],center[1]+radius[1],center[2]-radius[2]],
                                               [center[0]+radius[0],center[1]+radius[1],center[2]-radius[2]],
                                               [center[0]+radius[0],center[1]-radius[1],center[2]-radius[2]],
                                               [center[0]-radius[0],center[1]-radius[1],center[2]+radius[2]],
                                               [center[0]-radius[0],center[1]+radius[1],center[2]+radius[2]],
                                               [center[0]+radius[0],center[1]+radius[1],center[2]+radius[2]],
                                               [center[0]+radius[0],center[1]-radius[1],center[2]+radius[2]],
                                              ])
    line_indices = [[0, 1], [1, 2], [2, 3], [0, 3],
                    [4, 5], [5, 6], [6, 7], [7, 4],
                    [0, 4], [1, 5], [2, 6], [3, 7]]
    lines.lines = o3d.utility.Vector2iVector(line_indices)
    #lines.colors = o3d.utility.Vector3dVector(color_lines)
    vis.add_geometry(lines)

    # Draw point cloud.
    
    rays = get_rays_nsvf_np(H, W, intrinsic, extrinsics[i_test])

    for i_view in range(len(i_test)):
        rgb_i = rgbs[i_view].reshape(-1, 3)
        depth_i = depths[i_view].reshape(-1)
        rays_i = rays[i_view]
        
        origins_i = rays_i[:, 0]
        dirs_i = rays_i[:, 1]

        points = origins_i + depth_i[:, None] * dirs_i
        
        pointcloud_i = o3d.geometry.PointCloud()
        pointcloud_i.points = o3d.utility.Vector3dVector(points[::stride])
        pointcloud_i.colors = o3d.utility.Vector3dVector(rgb_i[::stride])

        vis.add_geometry(pointcloud_i)


    render_opt = vis.get_render_option()
    #render_opt.line_width = 10
    #render_opt.point_size = 10
    
    ctl = vis.get_view_control()
    ctl.set_constant_z_near(-10)
    ctl.rotate(x=0, y=-600)

    vis.capture_screen_image(os.path.join(img_dir, 'view_front.png'), do_render=True)
    
    ctl.rotate(x=-30, y=-350)
    vis.capture_screen_image(os.path.join(img_dir,'view_top.png'), do_render=True)
    
    vis.destroy_window()
    del vis

def camera_traj_visualization(intrinsic, extrinsics, traj_dense, H, W, img_name):
    if extrinsics.shape[-2:] == (3, 4):
        extrinsics = to_4x4(extrinsics)

    n_cams = extrinsics.shape[0]
    n_lines = traj_dense.shape[0]
    #n_controls = control_points.shape[0]
    
    color_cams = get_colormap(n_cams)
    color_lines = get_colormap(n_lines-1)
    #color_controls = get_colormap(n_controls)

    vis = o3d.visualization.Visualizer()

    vis.create_window(visible=False)
    
    # draw trajectory.
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(traj_dense)
    line_indices = [[i-1, i] for i in range(1, n_lines)]
    lines.lines = o3d.utility.Vector2iVector(line_indices)
    lines.colors = o3d.utility.Vector3dVector(color_lines)

    vis.add_geometry(lines)

    # draw control points.
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(control_points)
    cloud.colors = o3d.utility.Vector3dVector(color_controls)
    
    vis.add_geometry(cloud)
    """
    render_opt = vis.get_render_option()
    render_opt.line_width = 10
    render_opt.point_size = 10
    
    # draw camera.
    for i_cam, extrinsic_i in enumerate(extrinsics):
        # draw camera pose.
        cam_obj_i = create_camera_lineset(intrinsic, extrinsic_i, H, W, color_cams[i_cam])
        
        vis.add_geometry(cam_obj_i)

    ctl = vis.get_view_control()
    pinhole_init = o3d.camera.PinholeCameraParameters()
    ctl.set_constant_z_near(-10)
    ctl.rotate(x=-360, y=-360)
    
    #vis.run()
    vis.capture_screen_image(img_name, do_render=True)
    vis.destroy_window()
    del vis
    #
    #vis.draw(cam_obj_list, field_of_view=100.0, point_size=10, line_width=2, show_skybox=False)


def create_camera_lineset(intrinsic, extrinsic, H, W, color, scale=0.01):
    P0 = extrinsic[:3, 3]
    # points on image coordinates.
    p1 = [0, 0, 1]
    p2 = [W, 0, 1]
    p3 = [W, H, 1]
    p4 = [0, H, 1]

    points = np.stack([p1, p2, p3, p4], axis=-1).astype(np.float64)
    
    points_3d_cam = np.matmul(np.linalg.inv(intrinsic), points) * scale
    points_3d_world = np.matmul(extrinsic[:3, :3], points_3d_cam) + P0[:, None].repeat(4, 1)
    
    P_cam = np.concatenate([P0[:, None], points_3d_world], axis=-1).T
    
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(P_cam)
    
    line_indices = [[0,1], [0,2], [0,3], [0,4], [1,2], [2,3], [3,4], [4,1]]
    lines.lines = o3d.utility.Vector2iVector(line_indices)

    lines.paint_uniform_color(color)
    
    return lines
        



