import os

import torch
import numpy as np

from torchvision.models import squeezenet1_1

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config
from Utils_torch import im_to_numpy
from third.S2CRNet.models import S2CRNet
import torchvision.transforms as transforms


class Image_Harmonization(object):

    def __init__(self):
        size = (375, 1242)
        self.size = size
        self.lr_trans = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        path = config.common_config.image_harmonization_model_path
        model = S2CRNet(squeezenet1_1(pretrained=False), stack=True).to(device)
        model.load_state_dict(torch.load(path, map_location=device)['state_dict'])
        self.model = model

    def process_data(self, obj_rgb, obj_mask, bg_rgb, pos):
        bg_empty = np.zeros((bg_rgb.size[1], bg_rgb.size[0]))
        bg_empty = Image.fromarray(np.uint8(bg_empty)).convert('RGB')
        compose_rgb = bg_rgb.copy()
        compose_mask = bg_empty.copy()
        compose_rgb.paste(obj_rgb, pos, mask=obj_mask)
        compose_mask.paste(obj_mask, pos, mask=obj_mask)
        return compose_rgb, compose_mask

    def convert_data2tensor(self, img, mask):
        bbox = mask.getbbox()
        thumb_foreground = img.crop(bbox)
        thumb_mask = mask.crop(bbox)
        data = {"composite_images": torch.unsqueeze(self.lr_trans(img), 0),
                "mask": torch.unsqueeze(self.lr_trans(mask), 0),
                "fore_images": torch.unsqueeze(self.lr_trans(thumb_foreground), 0),
                "fore_mask": torch.unsqueeze(self.lr_trans(thumb_mask), 0),
                "label": torch.tensor([4]),
                "ori_img": torch.unsqueeze(self.lr_trans(img), 0),
                "real_images": torch.unsqueeze(self.lr_trans(img), 0),
                }
        return data

    def run(self, obj_rgb, obj_mask, bg_rgb, pos):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        obj_rgb = Image.fromarray(obj_rgb).convert('RGB')
        obj_mask = Image.fromarray(obj_mask).convert('L')
        bg_rgb = Image.fromarray(bg_rgb).convert('RGB')
        compose_rgb, compose_mask = self.process_data(obj_rgb, obj_mask, bg_rgb, pos)
        with torch.no_grad():
            batches = self.convert_data2tensor(compose_rgb, compose_mask)

            inputs = batches['composite_images'].to(device)
            target = batches['real_images'].to(device)
            mask = batches['mask'].to(device)
            fore = batches["fore_images"].to(device)
            foremask = batches["fore_mask"].to(device)
            label = batches['label']
            ori_img = batches["ori_img"].to(device)

            label_oh = torch.zeros(1, 5).scatter_(1, label.view(label.shape[0], 1), 1).to(
                torch.float32).to(device)
            feeded = torch.cat([inputs, mask], dim=1).to(device)
            fore = torch.cat([fore, foremask], dim=1).to(device)

            outputs, param1, param2 = self.model(ori_img, feeded, fore, label_oh, True)
            im_stage1 = outputs[1] * mask + target * (1 - mask)
            im_stage2 = outputs[0] * mask + target * (1 - mask)
            final_img = im_to_numpy(torch.clamp(im_stage2[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
            return final_img


if __name__ == '__main__':
    sub_dir = "000009"
    img_file = "theta_93.50_phi_3.15.png"
    bg_file = "000007.png"
    rp = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/image_harmonization/S2CRNet-demos/image/datasets/ShapeNet"
    img_dir = "{}/rgb/{}/".format(rp, sub_dir)
    img_path = os.path.join(img_dir, img_file)
    sub_dir = "{}/mask/{}/".format(rp, sub_dir)
    mask_path = os.path.join(sub_dir, img_file)
    bg_dir = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/image_harmonization/S2CRNet-demos/my_dataset/bg/"

    pos = 700, 190
    bg_path = os.path.join(bg_dir, bg_file)
    obj_rgb = Image.open(img_path).convert('RGB')
    obj_mask = Image.open(mask_path).convert('L')
    bg_rgb = Image.open(bg_path).convert('RGB')
    Image_Harmonization().run(obj_rgb, obj_mask, bg_rgb, pos)
