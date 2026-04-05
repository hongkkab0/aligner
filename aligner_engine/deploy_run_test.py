# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math

import cv2
import os

###################
from mmdeploy.apis import build_task_processor
from mmdeploy.utils.config_utils import load_config
import time
from aligner_engine.mm_deploy.dice.dice_rotated_detection import DiceRotatedDetection
import torch
from torch import Tensor
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    # parser.add_argument('device_name', help='name of device, cuda or cpu')
    # parser.add_argument(
    #     'model_path',
    #     help='path of mmdeploy SDK model dumped by model converter')
    # parser.add_argument('image_path', help='path of an image')

    parser.add_argument('deploy_cfg', help='')
    parser.add_argument('model_cfg', help='')
    parser.add_argument('model', help='~~.onnx')
    parser.add_argument('device', help='cuda:0 or cpu')
    parser.add_argument('image_path', help='path of an image')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg

    img = cv2.imread(args.image_path)

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
    # task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)
    # task_processor.build_backend_model =DiceRotatedDetection.build_backend_model
    task_processor = DiceRotatedDetection(model_cfg, deploy_cfg, args.device)
    model = task_processor.build_backend_model(
        [args.model],
        data_preprocessor_updater=task_processor.update_data_preprocessor)

    # build pipeline
    from mmrotate.utils import register_all_modules
    from mmdet.utils import get_test_pipeline_cfg
    from mmcv.transforms import Compose

    register_all_modules()
    test_pipeline = get_test_pipeline_cfg(model_cfg_path)
    test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    pipeline = Compose(test_pipeline)

    data_ = dict(img=img, img_id=0)
    data_ = pipeline(data_)
    data_['inputs'] = [data_['inputs']]
    data_['data_samples'] = [data_['data_samples']]

    for i in range(20):
        s = time.time()
        result = model.test_step(data_)[0]
        e = time.time()
        print(e-s)

    predictions = {}
    for pred_idx, pred in enumerate(result.pred_instances):
        # label = pred.labels.detach().cpu().item()
        # class_name = self._classes[label]
        class_name = "object"
        qbox = rbox2qbox(pred.bboxes.tensor)[0].detach().cpu().tolist()
        predictions[str(pred_idx)] = dict(
            class_name=class_name,
            conf=pred.scores.detach().cpu().item(),
            qbox=qbox)  # [x1 y1 x2 y2 x3 y3 x4 y4]

    for pred_key, pred in predictions.items():
        polygon_detached =np.array(pred['qbox']).reshape(-1,2)
        cv2.line(img,
                 (int(polygon_detached[0][0]), int(polygon_detached[0][1])),
                 (int(polygon_detached[1][0]), int(polygon_detached[1][1])),
                 (0, 255, 0), thickness=1)
        cv2.line(img,
                 (int(polygon_detached[1][0]), int(polygon_detached[1][1])),
                 (int(polygon_detached[2][0]), int(polygon_detached[2][1])),
                 (0, 255, 0), thickness=1)
        cv2.line(img,
                 (int(polygon_detached[2][0]), int(polygon_detached[2][1])),
                 (int(polygon_detached[3][0]), int(polygon_detached[3][1])),
                 (0, 255, 0), thickness=1)
        cv2.line(img,
                 (int(polygon_detached[3][0]), int(polygon_detached[3][1])),
                 (int(polygon_detached[0][0]), int(polygon_detached[0][1])),
                 (0, 255, 0), thickness=1)

    cv2.imwrite('output_detection_vino.png', img)


def rbox2qbox(boxes: Tensor) -> Tensor:
    """Convert rotated boxes to quadrilateral boxes.

    Args:
        boxes (Tensor): Rotated box tensor with shape of (..., 5).

    Returns:
        Tensor: Quadrilateral box tensor with shape of (..., 8).
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
        batch_size = 1

    pts = torch.empty(batch_size, 8).to(ctr.device)
    for i, [v1, v2, p1, p2, p3, p4] in enumerate(zip(vec1, vec2, pt1, pt2, pt3, pt4)):
        if v1[0] + v2[0] < 0 and v1[1] + v2[1] < 0:
            pts[i] = torch.cat([p1, p4, p3, p2], dim=-1)
        elif v1[0] + v2[0] >= 0 and v1[1] + v2[1] > 0:
            pts[i] = torch.cat([p3, p2, p1, p4], dim=-1)
        if v1[0] - v2[0] < 0 and v1[1] - v2[1] < 0:
            pts[i] = torch.cat([p2, p1, p4, p3], dim=-1)
        elif v1[0] - v2[0] >= 0 and v1[1] - v2[1] > 0:
            pts[i] = torch.cat([p4, p3, p2, p1], dim=-1)
    return pts


if __name__ == '__main__':
    main()
    # main_pth()
    pass
