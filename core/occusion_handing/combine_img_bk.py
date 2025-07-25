import math

import matplotlib.pyplot as plt

import config
from Utils_common import Get_List_Max_Index, get_mask_from_RGBA
import numpy as np
import cv2

from core.sensor_simulation.image_refine import Image_Harmonization


def combine_bg_with_obj(bg, objs_chaos, coordinates_chaos, centers):
    image_refine = Image_Harmonization()
    bg_ymax, bg_xmax, _ = bg.shape

    distance = []
    for cen in centers:
        distance.append(math.sqrt(cen[0] ** 2 + cen[1] ** 2))
    index = np.asarray(Get_List_Max_Index(distance, len(distance)))
    objs = []
    for ix in index:
        objs.append(objs_chaos[ix])
    coordinates = np.asarray(coordinates_chaos, dtype=object)[index]
    masks = []
    for obj in objs:
        mask = get_mask_from_RGBA(obj)
        masks.append(mask)

    bg_temp = bg.copy()
    for obj, coordinate, mask in zip(objs, coordinates, masks):
        obj_ymin, obj_ymax, obj_xmin, obj_xmax = coordinate
        for i in range(obj.shape[0]):
            for j in range(obj.shape[1]):

                if not obj[i][j][3] == 0 and 0 <= i + obj_ymin < bg_ymax and 0 <= j + obj_xmin < bg_xmax:
                    bg_temp[i + obj_ymin, j + obj_xmin, :3] = obj[i, j, :3]
                else:
                    pass
    spp = image_refine.size
    bg_temp = cv2.resize(bg_temp, (spp[1], spp[0]), interpolation=cv2.INTER_CUBIC)

    bg_temp2 = bg.copy()

    for obj, coordinate, mask in zip(objs, coordinates, masks):
        obj_ymin, obj_ymax, obj_xmin, obj_xmax = coordinate

        obj_rgb = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
        bg_temp2_rgb = cv2.cvtColor(bg_temp2, cv2.COLOR_BGR2RGB)
        bg_temp2 = image_refine.run(obj_rgb, mask, bg_temp2_rgb, (obj_xmin, obj_ymin))
        bg_temp2 = cv2.cvtColor(bg_temp2, cv2.COLOR_RGB2BGR)
    return bg_temp, bg_temp2
