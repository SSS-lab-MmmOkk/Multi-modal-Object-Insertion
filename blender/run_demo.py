'''
RENDERING PIPELINE DEMO
run it several times to see random images with different lighting conditions,
viewpoints, truncations and backgrounds.
'''

import os
import sys
from PIL import Image
import random
from datetime import datetime

random.seed(datetime.now())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
print(os.path.join(BASE_DIR, './'))
from blender.global_variables import *

debug_mode = 1

if debug_mode:
    io_redirect = ''
else:
    io_redirect = ' > /dev/null 2>&1'

syn_images_folder = os.path.join(BASE_DIR, 'demo_images')
model_name = 'Car_5'
image_name = 'demo_img_x.png'
if not os.path.exists(syn_images_folder):
    os.mkdir(syn_images_folder)
    os.mkdir(os.path.join(syn_images_folder, model_name))

viewpoint_samples_file = os.path.join(BASE_DIR, 'sample_viewpoints.txt')
viewpoint_samples = [[float(x) for x in line.rstrip().split(' ')] for line in open(viewpoint_samples_file, 'r')]

v = random.choice(viewpoint_samples)
print(">> Selected view: ", v)
python_cmd = 'python %s -a %s -e %s -t %s -d %s -o %s' % (os.path.join(BASE_DIR, 'render_class_view.py'),
                                                          str(v[0]), str(v[1]), str(v[2]), str(v[3]),
                                                          os.path.join(syn_images_folder, model_name, image_name))
print(">> Running rendering command: \n \t %s" % (python_cmd))
os.system('%s %s' % (python_cmd, io_redirect))

print(">> Displaying rendered image ...")
im = Image.open(os.path.join(syn_images_folder, model_name, image_name))
im.show()
