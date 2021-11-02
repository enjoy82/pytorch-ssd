import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


#image_size = 300
image_size = 600
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2
"""
specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]
"""
specs = [
    SSDSpec(40, 16, SSDBoxSizes(30, 120), [2, 3]),
    SSDSpec(20, 32, SSDBoxSizes(120, 210), [2, 3]),
    SSDSpec(10, 64, SSDBoxSizes(210, 300), [2, 3]),
    SSDSpec(8, 100, SSDBoxSizes(300, 390), [2, 3]),
    SSDSpec(4, 150, SSDBoxSizes(390, 480), [2, 3]),
    SSDSpec(2, 300, SSDBoxSizes(480, 570), [2, 3]),
    SSDSpec(1, 600, SSDBoxSizes(570, 660), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)