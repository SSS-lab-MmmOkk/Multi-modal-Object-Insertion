import importlib
m = importlib.import_module('core.occusion_handing.occ')
import math
import open3d as o3d
import numpy as np
import config
from Utils_common import Get_List_Max_Index, read_Bin_PC
from Utils_label import get_occlusion_level
from Utils_o3d import pc_numpy_2_o3d, pc_o3d_2_numpy, load_normalized_mesh_obj, convert_corner2o3d_box
from core.pose_estimulation.pose_generator import tranform_mesh_by_pose
from core.sensor_simulation.lidar_simulator import lidar_simulation
from utils import calibration_kitti, object3d_kitti, box_utils




def update_single_labels(label, info, delete_points_mask):
    count = np.sum(delete_points_mask[info[0]: info[0] + info[1]])

    occlusion_ratio = count / info[1]
    occlusion_level = get_occlusion_level(occlusion_ratio)
    label[2] = str(occlusion_level)
    if occlusion_ratio >= config.common_config.occlusion_th:
        label[0] = "DontCare"
    return label


def update_labels(labels, infos, delete_points_mask):
    labels_updated = []
    for index, (label, info) in enumerate(zip(labels, infos)):
        labels_updated.append(update_single_labels(label, info, delete_points_mask))
    return labels_updated


def get_distance(meshes):
    distance = []
    for obj in meshes:
        xyz = obj.get_center()
        r = math.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] + +2)
        distance.append(r)
    return distance


def extact_labels_from_bg(calib_path, label_path):
    calib_info = calibration_kitti.Calibration(calib_path)

    lines = []
    with open(label_path, 'r') as f:
        for line in f:
            lines.append(line.strip().split())
    lines = [line for line in lines if line[0] != 'DontCare']

    obj_list = object3d_kitti.get_objects_from_label(label_path)

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

    return corners_lidar, lines


def update_single_init_label(label, init_pc, combine_pc, corner):
    box = convert_corner2o3d_box(corner)
    indexesWithinBox_init = box.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(init_pc))
    indexesWithinBox = box.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(combine_pc))
    if len(indexesWithinBox_init) != 0:
        occlusion_ratio = len(indexesWithinBox_init) - len(indexesWithinBox) / len(indexesWithinBox_init)
    else:
        occlusion_ratio = 1
    occlusion_level = get_occlusion_level(occlusion_ratio)
    label[2] = str(occlusion_level)
    if occlusion_ratio >= config.common_config.occlusion_th or \
            len(indexesWithinBox) < config.common_config.occ_point_max:
        label[0] = "DontCare"
    return label


def update_init_label(labels, init_pc, combine_pc, corners):
    label_updated = []
    for label, corner in zip(labels, corners):
        label_updated.append(update_single_init_label(label, init_pc, combine_pc, corner))
    return label_updated


def combine_single_pcd(delete_points_mask, delete_mask, bg, obj, infos):
    if delete_points_mask is None:
        delete_points_mask = delete_mask
    else:
        delete_points_mask = np.logical_or(delete_points_mask, delete_mask)

    infos.append((len(bg), len(obj.points)))
    bg = np.concatenate((bg, np.asarray(obj.points)), axis=0)
    delete_points_mask = np.concatenate((delete_points_mask, np.zeros((len(obj.points),), dtype=bool)))
    return infos, bg, delete_points_mask


def combine_pcd(bg, pcds, meshes, labels_2):
    distance = get_distance(meshes)
    index = np.asarray(Get_List_Max_Index(distance, len(distance)))
    objs_order = np.asarray(pcds, dtype=object)[index]
    meshes_order = np.asarray(meshes, dtype=object)[index]
    labels_2_order = np.asarray(labels_2, dtype=object)[index]

    infos = []
    delete_points_mask = None

    for obj, mesh in zip(objs_order, meshes_order):
        delete_mask = m.get_delete_points_idx(mesh, bg)
        infos, bg, delete_points_mask = combine_single_pcd(delete_points_mask, delete_mask, bg, obj, infos)

    mask = ~delete_points_mask
    assert len(mask) == len(bg)
    bg = bg[mask]

    labels_2_update = update_labels(labels_2_order, infos, delete_points_mask)

    return bg, labels_2_update


if __name__ == "__main__":
    vis = o3d.visualization.Visualizer()

    bin_path = f"{config.common_config.project_dir}/_datasets/tiny_kitti/training/velodyne/000007.bin"

    calib_path = f"{config.common_config.project_dir}/_datasets/tiny_kitti/training/calib/000007.txt"
    label_path = f"{config.common_config.project_dir}/_datasets/tiny_kitti/training/label_2/000007.txt"
    bg_xyz = read_Bin_PC(bin_path)
    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(bg_xyz)

    obj_mesh_path = f"{config.common_config.project_dir}/_assets/tiny_shapenet/Car_2/models/model_normalized.obj"
    mesh_obj_initial = load_normalized_mesh_obj(obj_mesh_path)

    position, rz_degree = [12, -1, -0.8663082560743849], -5
    position2, rz_degree2 = [17, -3, -0.8663082560743849], -5
    position3, rz_degree3 = [17, -1, -0.8663082560743849], -5
    position4, rz_degree4 = [10, 5, -0.8663082560743849], -5

    car = tranform_mesh_by_pose(mesh_obj_initial, position, rz_degree)
    car2 = tranform_mesh_by_pose(mesh_obj_initial, position2, rz_degree2)
    car3 = tranform_mesh_by_pose(mesh_obj_initial, position3, rz_degree3)
    car4 = tranform_mesh_by_pose(mesh_obj_initial, position4, rz_degree4)

    pcd_obj = lidar_simulation(car)
    pcd_obj2 = lidar_simulation(car2)
    pcd_obj3 = lidar_simulation(car3)
    pcd_obj4 = lidar_simulation(car4)

    combine_pc, labels = combine_pcd(bg_xyz, [pcd_obj, pcd_obj2, pcd_obj3, pcd_obj4],
                                     [car, car2, car3, car4],
                                     [[1, 2, 0, 2], [1, 2, 0, 2], [1, 2, 0, 2],
                                      [1, 2, 0, 2]])

    corners, lines = extact_labels_from_bg(calib_path, label_path)
    bg_labels = update_init_label(lines, bg_xyz, combine_pc, corners)

    print(bg_labels)
    combine_pcd_o3d = pc_numpy_2_o3d(combine_pc)

    print(labels[0])
    print(labels)

    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()
    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)
    # vis.add_geometry(obj_shadow_mesh)
    vis.add_geometry(combine_pcd_o3d)
    vis.run()
    vis.destroy_window()
