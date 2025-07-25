from config.common_config import occlusion_th
from core.sensor_simulation.lidar_simulator import *


def get_occlusion_level(occlusion_ratio):
    return int(np.clip(occlusion_ratio * 4, 0, 3))


def get_care_labels(bg_labels, key=None):
    bg_labels_dont_care = []
    bg_labels_care = []
    for x in bg_labels:
        if "DontCare" in x:
            bg_labels_dont_care.append(x)
        else:
            if key is None:
                bg_labels_care.append(x)
            else:
                if key in x:
                    bg_labels_care.append(x)
                else:
                    bg_labels_dont_care.append(x)
    return bg_labels_care, bg_labels_dont_care


def sort_labels(labels_input):
    labels = labels_input.copy()
    if isinstance(labels, list):
        labels.sort(key=lambda x: x[0])
    else:
        labels = labels[np.argsort(labels[:, 0])]
    bg_labels_care, bg_labels_dont_care = get_care_labels(labels)
    labels = bg_labels_care + bg_labels_dont_care

    return np.array(labels)


def del_labels(bg_labels, keys):
    bg_labels_care = []
    for x in bg_labels:
        flag = True
        for key in keys:
            if key in x:
                flag = False
        if flag:
            bg_labels_care.append(x)
    return bg_labels_care


def update_bg_labels_care(bg_labels_care, bg_box_corners, pcd_bg, pcd_bg_update):
    from logger import CLogger
    bg_labels_update = []
    for bg_label, bg_box in zip(bg_labels_care, bg_box_corners):
        crop_box = o3d.geometry.OrientedBoundingBox. \
            create_from_points(o3d.utility.Vector3dVector(bg_box))
        nums1 = len(np.asarray(pcd_bg.crop(crop_box).points))
        nums2 = len(np.asarray(pcd_bg_update.crop(crop_box).points))
        if nums1 == 0:
            occlusion_ratio = 0
            CLogger.warning(f"zero points in bounding box, {bg_label}")
        else:
            occlusion_ratio = 1 - nums2 / nums1
            occlusion_level = get_occlusion_level(occlusion_ratio)

        if occlusion_ratio >= occlusion_th:
            bg_label[0] = "DontCare"
        bg_labels_update.append(bg_label)
    return bg_labels_update


def write_labels_2(path, labels):
    with open(path, 'w') as f:
        for label in labels:
            f.writelines(" ".join(label) + "\n")


def read_labels_2(path, ignore_type=None):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data_line = line.strip("\n").split()
            if ignore_type is not None:
                flag = True
                for key in ignore_type:
                    if key in data_line:
                        flag = False
                        break
                if flag:
                    data.append(data_line)
            else:
                data.append(data_line)
    return data
