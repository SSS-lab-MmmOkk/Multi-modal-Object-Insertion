import open3d as o3d

import config
import numpy as np


def pc_numpy_2_o3d(xyz):
    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(xyz)
    return pcd_bg


def pc_o3d_2_numpy(pcd_obj):
    return np.asarray(pcd_obj.points)


def convert_corner2o3d_box(corner):
    return o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corner))


def set_point_cloud_color(non_road_pc, colors):
    non_road_pc.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return non_road_pc


def load_normalized_mesh_obj(obj_mesh_path):
    mesh_obj = o3d.io.read_triangle_mesh(obj_mesh_path)
    obj_scale(mesh_obj, scale=config.common_config.multi_scale)

    return mesh_obj


def get_space_center(obj):
    xyz_min = obj.get_min_bound()
    xyz_max = obj.get_max_bound()
    differ = (xyz_max + xyz_min) / 2
    return differ


def obj_scale(obj, scale):
    obj.scale(scale, center=obj.get_center())

    R1 = obj.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    obj.rotate(R1)
    R2 = obj.get_rotation_matrix_from_xyz((0, 0, -np.pi / 2))
    obj.rotate(R2)


def create_o3d_box_from_points(box):
    box_o3d = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(box))
    return box_o3d
