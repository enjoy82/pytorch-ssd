from typing import List
import numpy as np




def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    #_, indexes = scores.sort(descending=True)
    indexes = np.argsort(scores)
    #indexes = indexes[:candidate_size]
    indexes = indexes[::-1]
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        #current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        #indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
        #TODO
    if nms_method == "soft":
        return hard_nms(box_scores, score_threshold, sigma, top_k)
    else:
        return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)

# def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
#         sigma=0.5, top_k=-1, candidate_size=200):
#     if nms_method == "soft":
#         return soft_nms(box_scores, score_threshold, sigma, top_k)
#     else:
#         return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)

#
# def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
#     """Soft NMS implementation.
#
#     References:
#         https://arxiv.org/abs/1704.04503
#         https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx
#
#     Args:
#         box_scores (N, 5): boxes in corner-form and probabilities.
#         score_threshold: boxes with scores less than value are not considered.
#         sigma: the parameter in score re-computation.
#             scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
#         top_k: keep top_k results. If k <= 0, keep all the results.
#     Returns:
#          picked_box_scores (K, 5): results of NMS.
#     """
#     picked_box_scores = []
#     while box_scores.size(0) > 0:
#         max_score_index = torch.argmax(box_scores[:, 4])
#         cur_box_prob = torch.tensor(box_scores[max_score_index, :])
#         picked_box_scores.append(cur_box_prob)
#         if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
#             break
#         cur_box = cur_box_prob[:-1]
#         box_scores[max_score_index, :] = box_scores[-1, :]
#         box_scores = box_scores[:-1, :]
#         ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
#         box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
#         box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
#     if len(picked_box_scores) > 0:
#         return torch.stack(picked_box_scores)
#     else:
#         return torch.tensor([])
