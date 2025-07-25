import importlib
m = importlib.import_module('core.sensor_simulation.sim')
import numpy as np
import config




def get_ray_args():
    vertical_resolution = config.lidar_config.vertical_resolution
    horizontal_resolution = config.lidar_config.horizontal_resolution
    horizontal_left = config.lidar_config.horizontal_left
    horizontal_right = config.lidar_config.horizontal_right
    vertical_down = config.lidar_config.vertical_down
    vertical_up = config.lidar_config.vertical_up
    return [horizontal_left, horizontal_right, vertical_down, vertical_up, horizontal_resolution, vertical_resolution]


def lidar_simulation(mesh_obj):
    rays_args = get_ray_args()
    rays_args = m.get_min_ray_args4render_by_obj(mesh_obj,
                                                 config.lidar_config.extend_range,
                                                 rays_args)

    render_args = [config.lidar_config.noise_average,
                   config.lidar_config.noise_variance,
                   config.lidar_config.noise_severity,
                   config.lidar_config.loss_rate]

    pcd_obj = m.get_obj_pcd(mesh_obj, rays_args, render_args)

    return pcd_obj


def _test_ref(point=(0, 0, 0)):
    alpha = 0
    input_ref = 0
    rmax = 200
    dR = 0.09

    x, y, z = point
    ran = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ref = input_ref * np.exp(-2 * alpha * ran)
    P0 = ref * np.exp(-2 * alpha * ran) / (ran ** 2)
    Pmin = 0.9 * rmax ** (-2)
    snr = P0 / Pmin
    sig = dR / np.sqrt(2 * snr)
    if P0 < Pmin:

        ...
    else:

        ...


def mix_pc(pcd_road, pcd_non_road, pc_objs):
    np_pcd_road = np.asarray(pcd_road.points)
    np_pcd_non_road = np.asarray(pcd_non_road.points)
    flag = True
    for pc_obj in pc_objs:
        if flag:
            np_pc_objs = np.asarray(pc_obj.points)
            flag = False
        else:
            np_pc_objs = np.concatenate([np_pc_objs, np.asarray(pc_obj.points)], axis=0)

    mixed_pc_three = np.concatenate([np_pcd_road, np_pcd_non_road, np_pc_objs], axis=0)

    return mixed_pc_three


def complet_pc(mixed_pc_three):
    assert mixed_pc_three.shape[1] == 3

    hang = mixed_pc_three.shape[0]
    b = np.zeros((hang, 1))
    mixed_pc = np.concatenate([mixed_pc_three, b], axis=1)
    return mixed_pc


if __name__ == '__main__':
    lidar_height = 1.73
    verticle_view = 26.8
    horizontal_view = 360
    beam_num = 64
    horizontal_resolution = 0.09
    max_verticle_view = 15
    min_verticle_view = -11.8
    verticle_resolution = verticle_view / beam_num
    print(verticle_resolution)
