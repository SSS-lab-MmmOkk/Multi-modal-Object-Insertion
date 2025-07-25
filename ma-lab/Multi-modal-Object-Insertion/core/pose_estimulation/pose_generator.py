import copy
import math
from Utils_common import get_geometric_info, calculate_z_from_plane
import numpy as np


def is_on_road(road_range, position, half_diagonal):
    road_minx, road_maxx, road_miny, road_maxy = road_range
    x, y, _ = position

    if x + half_diagonal > road_maxx or x - half_diagonal < road_minx or y + half_diagonal > road_maxy or y - half_diagonal < road_miny:
        return False
    else:
        return True


def get_valid_pints(calib_info, points):
    assert points.shape[1] == 3
    import config
    img_shape = (config.camera_config.img_height, config.camera_config.img_width)
    pts_img, pts_rect_depth = calib_info.lidar_to_img(points)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    pts_fov = points[pts_valid_flag]
    return pts_fov


def generate_pose(mesh_obj, road_pc_input, road_labels):
    half_diagonal, _, half_height = get_geometric_info(mesh_obj)
    road_pc = road_pc_input.copy()
    road_pc = road_pc[road_pc[:,
                      0] > 4]
    road_pc = road_pc[road_pc[:, 2] < 3]
    road_pc = road_pc[road_pc[:, 2] > -3]
    sample_index = np.random.randint(0, len(road_pc))
    x, y, z = road_pc[sample_index][:3]
    position = [x, y, z + half_height]
    rz_degree = np.random.randint(2, 10)

    if y < 0: rz_degree = -rz_degree
    return position, rz_degree


def tranform_mesh_by_pose(mesh_obj, position, rz_degree):
    mesh_obj = copy.deepcopy(mesh_obj)

    mesh_obj.translate(position)
    rz_radians = math.radians(rz_degree)
    RZ = mesh_obj.get_rotation_matrix_from_xyz((0, 0, -rz_radians))
    mesh_obj.rotate(RZ)
    return mesh_obj
