import numpy as np
from .box_utils import nms
from .data_preprocessing import PredictionTransform
from .misc import Timer


class Predictor:
    def __init__(self, openvinonet, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, input = "data", output = "out"):
        self.openvinonet = openvinonet
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        self.sigma = sigma
        self.timer = Timer()
        self.input = input
        self.output = output

    def predict(self, image, top_k=-1, prob_threshold=None):
        image = self.transform(image)
        height, width= image.shape[1:]

        image = image.transpose((2, 0, 1))    # HWC > CHW 
        image = np.expand_dims(image, axis=0) # 次元合せ 

        #推論
        res = self.openvinonet.infer(inputs={self.input: image})
        boxes = res[self.output[0]][0]
        scores = res[self.output[1]][0]
        #TODO check this algorithm

        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        #TODO
        print(boxes.shape, scores.shape)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, len(scores)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return [], [], []
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], picked_labels, picked_box_probs[:, 4]