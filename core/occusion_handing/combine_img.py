import math

import matplotlib.pyplot as plt

import config
from Utils_common import Get_List_Max_Index, get_mask_from_RGBA
import numpy as np
import cv2


def combine_bg_with_obj_single(center, obj, bg):
    bg_ymax, bg_xmax, _ = bg.shape
    obg_ymax, obg_xmax, _ = obj.shape
    obj_xmin = center[0] - int(0.5 * obg_xmax)
    obj_ymin = center[1] - int(0.5 * obg_ymax)
    bg_temp = bg.copy()
    for i in range(obj.shape[0]):
        for j in range(obj.shape[1]):
            if not obj[i][j][3] == 0 and obj[i][j][0] >= 3 and obj[i][j][1] >= 3 and obj[i][j][
                2] >= 3 and 0 <= i + obj_ymin < bg_ymax and 0 <= j + obj_xmin < bg_xmax:
                bg_temp[i + obj_ymin, j + obj_xmin, :3] = obj[i, j, :3]
            else:
                pass
    return bg_temp


def combine_bg_with_obj_single_refine(center, obj, bg, mask):
    from core.sensor_simulation.image_refine import Image_Harmonization
    image_refine = Image_Harmonization()
    bg_ymax, bg_xmax, _ = bg.shape
    obg_ymax, obg_xmax, _ = obj.shape
    image_refine.size = (bg_ymax, bg_xmax)
    assert bg_ymax < bg_xmax
    obj_xmin = center[0] - int(0.5 * obg_xmax)
    obj_ymin = center[1] - int(0.5 * obg_ymax)

    obj_rgb = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
    bg_temp2_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    bg_temp2 = image_refine.run(obj_rgb, mask, bg_temp2_rgb, (obj_xmin, obj_ymin))
    bg_temp2 = cv2.cvtColor(bg_temp2, cv2.COLOR_RGB2BGR)
    return bg_temp2


def combine_bg_with_obj(bg, objs_chaos, coordinates, centers, refine=True):
    bg_ymax, bg_xmax, _ = bg.shape

    distance = []
    for cen in centers:
        distance.append(math.sqrt(cen[0] ** 2 + cen[1] ** 2))

    index = np.asarray(Get_List_Max_Index(distance, len(distance)))

    objs = []
    for ix in index:
        objs.append(objs_chaos[ix])

    pos_images = np.asarray(coordinates, dtype=object)[index]

    masks = []
    for obj in objs:
        mask = get_mask_from_RGBA(obj)
        masks.append(mask)

    bg_no_refine = bg.copy()
    bg_refine = bg.copy()

    for obj, pos_image in zip(objs, pos_images):
        bg_no_refine = combine_bg_with_obj_single(pos_image, obj, bg_no_refine)

    if config.camera_config.is_image_refine and refine:
        for obj, pos_image in zip(objs, pos_images):
            mask = get_mask_from_RGBA(obj)
            bg_refine = combine_bg_with_obj_single_refine(pos_image, obj, bg_refine, mask)
    else:
        bg_refine = bg_no_refine
    return bg_no_refine, bg_refine
