import os
import shutil

import numpy as np
import open3d as o3d
import config.common_config
from Utils_o3d import pc_numpy_2_o3d
from core.sensor_simulation.lidar_simulator import complet_pc


def load_road_split_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
    return labels


def split_pc(labels):
    inx_road_arr = []
    inx_other_road_arr = []
    inx_other_ground_arr = []
    inx_no_road_arr = []
    for i in range(len(labels)):
        lb = labels[i][0]
        if lb == 40:
            inx_road_arr.append(i)
        elif lb == 44:
            inx_other_road_arr.append(i)
        elif lb == 48:
            inx_other_road_arr.append(i)
        elif lb in (49, 70, 71, 72):
            inx_other_ground_arr.append(i)
        else:
            inx_no_road_arr.append(i)
    return inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr


def road_split(bg_index, bg_pc_path, save_road_dir, log_dir):
    dis_th = 5

    log_file = f"{log_dir}/road_split.log"
    road_split_pc_dir = config.common_config.road_split_pc_dir
    if os.path.exists(road_split_pc_dir):
        shutil.rmtree(road_split_pc_dir)
    os.makedirs(road_split_pc_dir, exist_ok=True)
    road_split_label_dir = config.common_config.road_split_label_dir
    pc_path = f"{road_split_pc_dir}/{bg_index:06d}.bin"
    label_path = f"{road_split_label_dir}/{bg_index:06d}.label"
    save_road_label_path = f"{save_road_dir}/{bg_index:06d}.label"
    save_road_interpolation_path = f"{save_road_dir}/{bg_index:06d}.bin"
    if os.path.exists(save_road_interpolation_path):
        labels = load_road_split_labels(save_road_label_path)
        road_pc = np.fromfile(save_road_interpolation_path, dtype=np.float32).reshape((-1, 3))
        inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr = split_pc(labels)
        pc = np.fromfile(bg_pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        _pc_non_road = pc[inx_other_road_arr + inx_other_ground_arr + inx_no_road_arr]
    else:
        print("split road .........")

        shutil.copyfile(bg_pc_path, pc_path)

        cmd1 = "cd ./third/CENet"
        cmd2 = f" python infer.py -d ./data -l ./result -m ./model/512-594 -s valid/test"

        os.system(f"{cmd1} && {cmd2} > {log_file} 2>&1")

        shutil.copyfile(label_path, save_road_label_path)

        labels = load_road_split_labels(save_road_label_path)

        inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr = split_pc(labels)
        if len(inx_road_arr) <= 10:
            return None, None, None, None

        pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        _pc_road, _pc_other_road, _pc_other_ground, _pc_no_road = \
            pc[inx_road_arr], pc[inx_other_road_arr], pc[inx_other_ground_arr], pc[inx_no_road_arr]
        _pc_non_road = pc[inx_other_road_arr + inx_other_ground_arr + inx_no_road_arr]
        pcd_road = pc_numpy_2_o3d(_pc_road)

        cl, ind = pcd_road.remove_radius_outlier(nb_points=7, radius=1)
        pcd_inlier_road = pcd_road.select_by_index(ind)

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_inlier_road, 10)

        pcd_inter = mesh.sample_points_uniformly(number_of_points=50000)

        _pc_inter = np.asarray(pcd_inter.points)
        dis = np.linalg.norm(_pc_inter, axis=1, ord=2)
        _pc_inter_valid = _pc_inter[dis > 4]

        road_pc = _pc_inter_valid.astype(np.float32)
        if dis_th is not None:
            road_pc = road_pc[road_pc[:, 0] > dis_th]
        road_pc.astype(np.float32).tofile(save_road_interpolation_path, )
    return road_pc, labels, _pc_non_road, inx_road_arr


if __name__ == '__main__':
    ...
