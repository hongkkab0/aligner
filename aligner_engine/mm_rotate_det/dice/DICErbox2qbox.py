import torch
from torch import Tensor
import cv2
import numpy as np
import math

def dice_rbox_to_qbox_single_with_sort_rule(boxes: Tensor, sort_rule = -1):
    """Convert rotated boxes to quadrilateral boxes
    """
    ctr, w, h, theta = torch.split(boxes, (2, 1, 1, 1), dim=-1)
    cos_value, sin_value = torch.cos(theta), torch.sin(theta)
    vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
    vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)

    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2

    batch_size = ctr.size()[0]
    if len(pt1.shape) == 1:
        pt1 = pt1.unsqueeze(0)
        pt2 = pt2.unsqueeze(0)
        pt3 = pt3.unsqueeze(0)
        pt4 = pt4.unsqueeze(0)

        vec1 = vec1.unsqueeze(0)
        vec2 = vec2.unsqueeze(0)
        theta = theta.unsqueeze(0)
        batch_size = 1

    if len(ctr.shape) == 2:
        ctr = ctr.squeeze()

    pts = torch.empty(batch_size, 8).to(ctr.device)
    for i, [v1, v2, p1, p2, p3, p4, theta_] in enumerate(zip(vec1, vec2, pt1, pt2, pt3, pt4, theta)):
        if theta_ <= 0:
            angle = theta_ + math.pi /2
        else:
            angle = theta_
        angle = angle * 180.0 / math.pi

        if sort_rule == -1:
            if 1.0 < angle < 45.0:  # y가 가장 작은 것
                sort_rule = 0
            elif 45.0 <= angle < 89.0:  # x 가 가장 작은 것
                sort_rule = 1
            else:
                sort_rule = 2

        if sort_rule == 0:   # y가 가장 작은 것
            min_idx = torch.argmin(torch.stack([p1[1], p2[1], p3[1], p4[1]]))
            if min_idx == 0:
                pts[i] = torch.cat([p1, p4, p3, p2], dim=-1)
            elif min_idx == 1:
                pts[i] = torch.cat([p2, p1, p4, p3], dim=-1)
            elif min_idx == 2:
                pts[i] = torch.cat([p3, p2, p1, p4], dim=-1)
            elif min_idx == 3:
                pts[i] = torch.cat([p4, p3, p2, p1], dim=-1)
        elif sort_rule == 1:   # x 가 가장 작은 것
            min_idx = torch.argmin(torch.stack([p1[0], p2[0], p3[0], p4[0]]))
            if min_idx == 0:
                pts[i] = torch.cat([p1, p4, p3, p2], dim=-1)
            elif min_idx == 1:
                pts[i] = torch.cat([p2, p1, p4, p3], dim=-1)
            elif min_idx == 2:
                pts[i] = torch.cat([p3, p2, p1, p4], dim=-1)
            elif min_idx == 3:
                pts[i] = torch.cat([p4, p3, p2, p1], dim=-1)
        else:
            p1_ctr_diff = p1 - ctr
            if p1_ctr_diff[0] <= 0 and p1_ctr_diff[1] <= 0:
                pts[i] = torch.cat([p1, p4, p3, p2], dim=-1)
            elif p1_ctr_diff[0] > 0 and p1_ctr_diff[1] <= 0:
                pts[i] = torch.cat([p2, p1, p4, p3], dim=-1)
            elif p1_ctr_diff[0] > 0 and p1_ctr_diff[1] > 0:
                pts[i] = torch.cat([p3, p2, p1, p4], dim=-1)
            elif p1_ctr_diff[0] <= 0 and p1_ctr_diff[1] > 0:
                pts[i] = torch.cat([p4, p3, p2, p1], dim=-1)

    return pts[0], sort_rule

