import os

import config
from Utils_common import change_3dbox, xyz_to_degree
from logger import CLogger
import cv2
import numpy as np


def camera_simulation(objs_dir, position_lidar, rz_degree, obj_mesh_path, data_obj, mesh_obj, calib_info, save_log_dir,
                      bg_calib_path):
    position_camera = calib_info.lidar_to_rect(np.asarray([position_lidar]))[0]
    BASE_DIR = config.common_config.project_dir
    filename = "%s_%s_%s_%s_self%s.png" % (
        data_obj, position_camera[0], position_camera[1], position_camera[2], rz_degree)
    python_cmd = 'python %s -m %s -x %s -y %s -z %s -o %s -l %s -c %s -r %s' % (
        os.path.join(BASE_DIR, 'blender/render_class_view.py'), obj_mesh_path,
        str(position_camera[0]), str(position_camera[1]), str(position_camera[2]),
        os.path.join(objs_dir, filename), save_log_dir, bg_calib_path, rz_degree)
    CLogger.info(">> Running rendering command: \n \t %s" % (python_cmd))
    os.system('%s' % (python_cmd))
    try:
        assert os.path.exists(os.path.join(objs_dir, filename))
    except:
        print(os.path.join(objs_dir, filename))
        assert 1 == 2
    img_obj = cv2.imread(os.path.join(objs_dir, filename), cv2.IMREAD_UNCHANGED)

    img_obj_crop = crop_obj_from_image(img_obj)
    return img_obj_crop


def crop_obj_from_image(img_obj):
    x, y, channels = img_obj.shape

    if np.sum(img_obj[:, :, 3] != 0) == 0:
        img_obj[0, 0, 3] = 1
        CLogger.debug(f"obj not in camera field {(x, y, channels)}")

    temp = []

    for i in range(x):

        x_flag = False
        for j in range(y):
            if not img_obj[i][j][3] == 0:
                x_flag = True
                break
        if x_flag:
            temp.append(img_obj[i, :, :4])
    temp1 = np.asarray(temp)
    x, y, channels = temp1.shape

    new_obj = []

    for j in range(y):

        y_flag = False
        for i in range(x):
            if not temp1[i][j][3] == 0:
                y_flag = True
                break
        if y_flag:

            if len(new_obj) == 0:
                new_obj.append(temp1[:, j, :4])
                new_obj = np.asarray(new_obj).reshape((-1, 1, 4))
            else:
                new_obj = np.concatenate([new_obj, temp1[:, j, :4].reshape((-1, 1, 4))], axis=1)

    return new_obj


def get_img_scale_ratio(img_range, img_obj_min_box):
    h, w, _ = img_obj_min_box.shape

    img_xmin, img_ymin = [int(x) for x in np.min(img_range, axis=0)]
    img_xmax, img_ymax = [int(x) for x in np.max(img_range, axis=0)]

    h_bg = img_ymax - img_ymin
    w_bg = img_xmax - img_xmin
    fh = h_bg / h
    fw = w_bg / w

    if fh > fw:
        alpha = fw

        after_h = h * alpha
        img_ymin = int(img_ymin + (h_bg - after_h) / 2)
        img_ymax = int(img_ymax - (h_bg - after_h) / 2)
    else:
        alpha = fh

        after_w = w * alpha
        img_xmin = int(img_xmin + (w_bg - after_w) / 2)
        img_xmax = int(img_xmax - (w_bg - after_w) / 2)

    coord = [img_ymin, img_ymax, img_xmin, img_xmax]
    CLogger.debug("insert coordinates:{}--[img_ymin, img_ymax, img_xmin, img_xmax]".format(coord))

    return alpha, coord
