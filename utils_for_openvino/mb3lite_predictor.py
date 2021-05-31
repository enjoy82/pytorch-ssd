import numpy as np
from .predictor import Predictor

#hyper-parameter must understand
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

def create_mobilenetv3_small_ssd_lite_predictor(openvinonet, candidate_size=200,image_size = 300, nms_method=None, sigma=0.5, input = "data", output = []):
    predictor = Predictor(openvinonet, image_size, image_mean,
                          image_std,
                          nms_method=nms_method,
                          iou_threshold=iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          input = input,
                          output = output)
    return predictor
