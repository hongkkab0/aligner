from mmcv.transforms import BaseTransform
from mmrotate.registry import TRANSFORMS
import cv2
import copy
import matplotlib.pyplot as plt
import mmcv


@TRANSFORMS.register_module()
class DiceShowImage(BaseTransform):
    """
    Save Image to debug
    """

    def __init__(self) -> None:
        pass

    def transform(self, results: dict) -> dict:
        bboxes = results["gt_bboxes"].convert_to('qbox')

        polygons = bboxes.convert_to('qbox').tensor
        polygons = polygons.reshape(-1, 4, 2)
        polygons = [p for p in polygons]
        img_transformed = copy.deepcopy(results['img'])
        for polygon in polygons:
            polygon_detached = polygon.detach().cpu().tolist()
            cv2.line(img_transformed,
                     (int(polygon_detached[0][0]), int(polygon_detached[0][1])),
                     (int(polygon_detached[1][0]), int(polygon_detached[1][1])),
                     (0, 255, 0), thickness=1)
            cv2.line(img_transformed,
                     (int(polygon_detached[1][0]), int(polygon_detached[1][1])),
                     (int(polygon_detached[2][0]), int(polygon_detached[2][1])),
                     (0, 255, 0), thickness=1)
            cv2.line(img_transformed,
                     (int(polygon_detached[2][0]), int(polygon_detached[2][1])),
                     (int(polygon_detached[3][0]), int(polygon_detached[3][1])),
                     (0, 255, 0), thickness=1)
            cv2.line(img_transformed,
                     (int(polygon_detached[3][0]), int(polygon_detached[3][1])),
                     (int(polygon_detached[0][0]), int(polygon_detached[0][1])),
                     (0, 255, 0), thickness=1)

        img_transformed = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB)

        img_original = mmcv.imread(results['img_path'], channel_order='rgb')
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_original)
        plt.title('original')

        plt.subplot(1, 2, 2)
        plt.imshow(img_transformed)
        plt.title('transformed')
        plt.show()

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
