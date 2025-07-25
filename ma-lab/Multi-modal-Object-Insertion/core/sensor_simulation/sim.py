import open3d as o3d
import numpy as np
import math
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
import config
import os


def get_rays(horizontal_left,
             horizontal_right,
             vertical_down,
             vertical_up,
             horizontal_resolution,
             vertical_resolution,
             ):
    points = ray_direction(horizontal_left,
                           horizontal_right,
                           vertical_down,
                           vertical_up,
                           horizontal_resolution,
                           vertical_resolution,
                           config.lidar_config.r)

    rays = create_rays(config.lidar_config.lidar_position, points)

    return rays


def create_rays(lidar_position, point_directions):
    assert len(lidar_position) == 3
    rays = []
    for point_direction in point_directions:
        ray = (lidar_position[0], lidar_position[1], lidar_position[2],
               point_direction[0], point_direction[1], point_direction[2])
        rays.append(ray)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    return rays


def ray_direction(horizontal_left,
                  horizontal_right,
                  vertical_down,
                  vertical_up,
                  horizontal_resolution,
                  vertical_resolution,
                  r):
    points_list = []

    circle_num = int((vertical_up - vertical_down) // vertical_resolution)

    for i in range(circle_num):
        degree = vertical_resolution * i + vertical_down
        rad_phi = degree * np.pi / 180
        pts = ray_direction_circle_simulation(horizontal_left, horizontal_right, horizontal_resolution, r, rad_phi)
        points_list += pts

    return points_list


def ray_direction_circle_simulation(horizontal_left,
                                    horizontal_right,
                                    horizontal_resolution,
                                    r,
                                    rad_phi):
    laster_num = int((horizontal_right - horizontal_left) // horizontal_resolution)

    points_list = []

    for i in range(laster_num):
        degree = horizontal_resolution * i + horizontal_left
        rad_theta = degree * np.pi / 180

        if rad_phi > 90 * np.pi / 180:
            rad_phi = 90 * np.pi / 180
        elif rad_phi < -90 * np.pi / 180:
            rad_phi = -90 * np.pi / 180
        if rad_theta > 90 * np.pi / 180:
            rad_theta = 90 * np.pi / 180
        elif rad_theta < -90 * np.pi / 180:
            rad_theta = -90 * np.pi / 180

        x, y, z = spherical_to_cartesian(r, rad_phi, rad_theta)
        points_list.append((x, y, z))

    return points_list


def get_min_ray_args4render_by_obj(obj, extend_range, rays_args):
    horizontal_left, horizontal_right = rays_args[0], rays_args[1]
    vertical_down, vertical_up = rays_args[2], rays_args[3]
    horizontal_resolution, vertical_resolution = rays_args[4], rays_args[5]

    box_points = obj.get_oriented_bounding_box().get_box_points()

    temp = np.asarray(box_points)
    for point in temp:
        _, latitude, longitude = cartesian_to_spherical(*list(point))

        latitude, longitude = latitude.value, longitude.value

        if longitude > np.pi: longitude = -(np.pi * 2 - longitude)

        latitude = math.degrees(latitude)
        longitude = math.degrees(longitude)

        if horizontal_left > longitude: horizontal_left = longitude
        if horizontal_right < longitude: horizontal_right = longitude
        if vertical_down > latitude: vertical_down = latitude
        if vertical_up < latitude: vertical_up = latitude

    horizontal_left -= extend_range
    horizontal_right += extend_range
    vertical_down -= extend_range
    vertical_up += extend_range

    return [horizontal_left, horizontal_right, vertical_down, vertical_up, horizontal_resolution, vertical_resolution]


def render_pcd(pointcloud_xyz, average, variance, severity, loss_rate):
    try:
        row, column = pointcloud_xyz.shape
    except:
        print(pointcloud_xyz)
        print(type(pointcloud_xyz))
        raise ValueError()
    jitter = np.random.normal(average, variance, size=(row, column)) * severity
    new_pc_xyz = (pointcloud_xyz + jitter).astype('float32')

    index = np.random.choice(row, size=int(row * (1 - loss_rate)), replace=False)
    return new_pc_xyz[index]


def get_obj_pcd(car, rays_args, render_args):
    car_t = o3d.t.geometry.TriangleMesh.from_legacy(car)
    car.paint_uniform_color([1, 1, 0])

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(car_t)

    rays = get_rays(*rays_args)

    ans = scene.cast_rays(rays)
    distance = ans["t_hit"].numpy()

    xyz_direction = rays.numpy()[:, 3:]

    xyz_position = []

    for i in range(len(xyz_direction)):
        r, phi, theta = cartesian_to_spherical(*list(xyz_direction[i, :]))
        if distance[i] == np.inf:
            pass
        else:
            x, y, z = spherical_to_cartesian(distance[i], phi, theta)
            xyz_position.append([x, y, z])

    points_obj = render_pcd(np.array(xyz_position), *render_args)

    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(points_obj)

    return pcd_obj
