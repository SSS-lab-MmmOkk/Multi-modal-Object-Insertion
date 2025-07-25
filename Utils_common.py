import copy
import math
import os.path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as RR

from Utils_label import get_occlusion_level

from utils import calibration_kitti, object3d_kitti, box_utils
import config
from utils.box_utils import iou_2d
import open3d as o3d


def get_mask_from_RGBA(rgb):
    mask = (rgb[:, :, 3] != 0).astype("uint8") * 255
    return mask


def read_Bin_PC(path, retrun_r=False):
    if path.split(".")[-1] == "npy":
        example = np.load(path).astype(np.float32)
    else:
        example = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    example_xyz = example[:, :3]

    if retrun_r:
        return example
    return example_xyz


def calculate_z_from_plane(addx, addy, plane):
    A = plane[0]
    B = plane[1]
    C = plane[2]
    D = plane[3]

    z = -1 * (A * addx + B * addy + D) / (C)

    return z


def extact_initial_objs_from_bg(calib_info, label_path, pc_path, ignore=True):
    obj_list = object3d_kitti.get_objects_from_label(label_path)

    if ignore:
        obj_list = [obj for obj in obj_list if obj.cls_type != 'DontCare']

    info = {}

    info['name'] = np.array([obj.cls_type for obj in obj_list])

    num_objects = len(obj_list)
    info['num_objs'] = num_objects

    loc = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    loc_lidar = calib_info.rect_to_lidar(loc)

    dims = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])

    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]

    loc_lidar[:, 2] += h[:, 0] / 2

    rots = np.array([obj.ry for obj in obj_list])
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)

    info['gt_boxes_lidar'] = gt_boxes_lidar

    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
    info['corners_lidar'] = corners_lidar

    return info


def get_geometric_info(obj):
    min_xyz = obj.get_min_bound()
    max_xyz = obj.get_max_bound()
    x_min, x_max = min_xyz[0], max_xyz[0]
    y_min, y_max = min_xyz[1], max_xyz[1]
    z_min, z_max = min_xyz[2], max_xyz[2]
    half_diagonal = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) / 2
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    half_height = (max_xyz[2] - min_xyz[2]) / 2

    return half_diagonal, center, half_height


def get_initial_box3d_in_bg(initial_corners):
    initial_boxes = []
    objs_half_diagonal = []
    objs_center = []

    for corners in initial_corners:
        x_min, y_min, z_min = np.min(corners, axis=0)
        x_max, y_max, z_max = np.max(corners, axis=0)

        half_diagonal = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) / 2

        center = [(x_min + x_max) / 2, (y_min + y_max) / 2]

        initial_boxes.append(get_initial_box3d(corners))
        objs_half_diagonal.append(half_diagonal)
        objs_center.append(center)

    return initial_boxes, objs_half_diagonal, objs_center


def get_initial_box3d(coners):
    """
    :param coners:
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2
    :return:
    """
    points_box = coners
    lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7],
                          [4, 5], [5, 6], [6, 7], [7, 4]])

    colors = np.array([config.lidar_config.initial_box_color for _ in range(len(lines_box))])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_box)
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def xyz_to_radians(x, y, z):
    theta_radians = math.atan2(y, x)
    phi_radians = math.atan2(z, math.sqrt(x ** 2 + y ** 2))
    return theta_radians, phi_radians


def xyz_to_degree(x, y, z):
    theta_radians = math.atan2(y, x)
    phi_radians = math.atan2(z, math.sqrt(x ** 2 + y ** 2))

    return math.degrees(theta_radians), math.degrees(phi_radians)


def Get_List_Max_Index(list_, n):
    N_large = pd.DataFrame({'score': list_}).sort_values(by='score', ascending=[False])
    return list(N_large.index)[:n]


def get_euler_from_matrix(R):
    euler_type = "XYZ"

    sciangle_0, sciangle_1, sciangle_2 = RR.from_matrix(R).as_euler(euler_type)

    return sciangle_0, sciangle_1, sciangle_2


def get_box3d_R(box3d):
    return copy.copy(box3d.R)


def change_3dbox(box3d):
    sciangle_0, sciangle_1, sciangle_2 = get_euler_from_matrix(get_box3d_R(box3d))

    if 2.5 > abs(sciangle_0) > 1.47:
        ...

    if abs(sciangle_0) > 2.5 or abs(sciangle_1) > 3 or abs(sciangle_2) > 3:

        R_return = box3d.get_rotation_matrix_from_zyx([-sciangle_2, -sciangle_1, - sciangle_0])

        box3d.rotate(R_return)
        R_return_1 = box3d.get_rotation_matrix_from_xyz([0, 0, -sciangle_2])
        box3d.rotate(R_return_1)


    else:
        R_return = box3d.get_rotation_matrix_from_zyx([-sciangle_2, -sciangle_1, - sciangle_0])
        box3d.rotate(R_return)
        R_return_1 = box3d.get_rotation_matrix_from_xyz([0, 0, sciangle_2])
        box3d.rotate(R_return_1)

    return box3d, [sciangle_0, sciangle_1, sciangle_2]


def get_labels(rz_degree, lidar_box, calib_info, image_box, truncation_ratio):
    place_holder = -1111
    label_2_prefix = ["Car", "0.00", "0", "-10000"]
    img_xmin, img_ymin, img_xmax, img_ymax = place_holder, place_holder, place_holder, place_holder
    if image_box is not None:
        img_xmin, img_ymin, img_xmax, img_ymax = image_box

    if truncation_ratio is not None:
        label_2_prefix[1] = str(round(truncation_ratio, 2))

    x_, y_, z_ = np.asarray(lidar_box.extent)
    h, w, l = z_, y_, x_

    corners = np.asarray(lidar_box.get_box_points())
    x_min, y_min, z_min = np.min(corners, axis=0)
    x_max, y_max, z_max = np.max(corners, axis=0)
    lidar_bottom_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]).reshape((1, 3))

    rect_bottom_center = calib_info.lidar_to_rect(lidar_bottom_center)[0]
    x, y, z = rect_bottom_center

    r_y = math.radians(rz_degree) + np.pi / 2
    paras = [img_xmin, img_ymin, img_xmax, img_ymax, h, w, l, x, y, z, r_y]
    label_2_suffix = [str(round(para, 2)) for para in paras]
    label_2_prefix.extend(label_2_suffix)
    return label_2_prefix


def get_labels_by_box(box, rz_degree, calib_info, coordinate):
    label_2_prefix = ["Car", "0.00", "0", "-10000"]
    corners = np.asarray(box.get_box_points())
    x_min, y_min, z_min = np.min(corners, axis=0)
    x_max, y_max, z_max = np.max(corners, axis=0)

    if coordinate is not None:
        truncation_ratio = get_truncation_ratio(coordinate)
        label_2_prefix[1] = str(round(truncation_ratio, 2))

    img_range, _ = calib_info.lidar_to_img(corners)
    img_xmin, img_ymin = np.min(img_range, axis=0)
    img_xmax, img_ymax = np.max(img_range, axis=0)

    x_, y_, z_ = np.asarray(box.extent)
    h, w, l = z_, y_, x_

    lidar_bottom_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]).reshape((1, 3))

    rect_bottom_center = calib_info.lidar_to_rect(lidar_bottom_center)[0]

    x, y, z = rect_bottom_center

    r_y = math.radians(rz_degree) + np.pi / 2

    paras = [img_xmin, img_ymin, img_xmax, img_ymax, h, w, l, x, y, z, r_y]
    label_2_suffix = [str(round(para, 2)) for para in paras]
    label_2_prefix.extend(label_2_suffix)
    return label_2_prefix


from shapely.geometry import box


def update_occ_only_image(labels):
    labels = labels.copy()
    from utils.object3d_kitti import Object3d
    label_objects = [Object3d(label=label) for label in labels]
    _, _, dis_arr, image_box_insert = get_objs_attr(label_objects, True)
    index = np.asarray(Get_List_Max_Index(dis_arr, len(dis_arr)))
    labels_order = []
    image_box_order = []
    for i in index:
        labels_order.append(list(labels[i]))
        image_box_order.append(image_box_insert[i])
    for i in range(len(image_box_order)):
        _box1 = np.array(image_box_order[i])
        _boxes2 = np.array(image_box_order[i + 1:])
        occlusion_ratio = get_image_occuliton_ratio(_box1, _boxes2)
        occlusion_level = get_occlusion_level(occlusion_ratio)
        labels_order[i][2] = occlusion_level
        if occlusion_ratio >= config.common_config.occlusion_th:
            labels_order[i][0] = "DontCare"

    labels = labels_order
    return labels


def get_image_occuliton_ratio(box1, boxes2):
    if len(boxes2) == 0:
        return 0
    box_union = None
    box1 = box(*box1)
    for box2 in boxes2:
        box2 = box(*box2)
        box_inter = box1.intersection(box2)
        if box_union is None:
            box_union = box_inter
        else:
            box_union = box_union.union(box_inter)
    return box_union.area / box1.area


def get_truncation_ratio(box1, box2):
    insert_xmin, insert_ymin, insert_xmax, insert_ymax = box1
    bg_size = (insert_ymax - insert_ymin) * (insert_xmax - insert_xmin)
    from collections import namedtuple

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

    obj = Rectangle(insert_xmin, insert_ymin, insert_xmax, insert_ymax)
    bg = Rectangle(box2[0], box2[1], box2[2], box2[3])

    area = overlapping_area(obj, bg)
    if area is not None:
        return (bg_size - area) / bg_size
    else:

        return 0


def overlapping_area(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx > 0) and (dy > 0):
        return dx * dy
    else:
        return None


if __name__ == '__main__':
    box1 = [0, 0, 2, 2]
    box2 = [-1, -1, 1, 1]
    box3 = [1, 1, 2, 2]
    boxes2 = [box2, box3]
    box1 = np.array(box1)
    boxes2 = np.array(boxes2)
    r = get_image_occuliton_ratio(box1, boxes2)
    print(r)
