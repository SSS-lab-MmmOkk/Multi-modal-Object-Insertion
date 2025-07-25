import math

import config
from Utils_common import get_euler_from_matrix, get_box3d_R
from Utils_o3d import pc_numpy_2_o3d
from logger import CLogger


def is_on_road(mesh_obj, road_pc, non_road_pc):
    box = mesh_obj.get_oriented_bounding_box()
    non_road_pcd = pc_numpy_2_o3d(non_road_pc)
    non_road_pcd_contained = non_road_pcd.crop(box)

    if len(non_road_pcd_contained.points) >= 3:
        return False
    else:
        return True


def is_occlusion_initail_obj(xy, half_diagonal, centers, diagonals):
    for k in range(len(diagonals)):

        y_dis = abs(centers[k][1] - xy[1])

        initial_obj_dis = math.sqrt(centers[k][0] ** 2 + centers[k][1] ** 2)
        insert_obj_dis = math.sqrt(xy[0] ** 2 + xy[1] ** 2)

        length = diagonals[k] + half_diagonal - 2

        if insert_obj_dis > initial_obj_dis and y_dis < length:
            return True

    return False


def is_3d_box_overlaped(diagonals, centers, half_diagonal, xy):
    if len(diagonals) == 0:
        return False
    for k in range(len(diagonals)):

        x_dis, y_dis = centers[k][0] - xy[0], centers[k][1] - xy[1]
        dis = math.sqrt(x_dis ** 2 + y_dis ** 2)

        length = diagonals[k] + half_diagonal

        if dis > length:
            pass
        else:

            return True
    return False


def collision_detection(xy, half_diagonal, objs_half_diagonal, objs_center, initial_box_num):
    occlusion_flag = config.not_behind_initial_obj and is_occlusion_initail_obj(xy, half_diagonal,
                                                                                objs_center[0:initial_box_num],
                                                                                objs_half_diagonal[0:initial_box_num])

    overlap_flag = is_3d_box_overlaped(objs_half_diagonal, objs_center, half_diagonal, xy)

    return (not overlap_flag) and (not occlusion_flag)


def is_valid_pc_box(box_o3d):
    return abs(get_euler_from_matrix(get_box3d_R(box_o3d))[0]) > 3
