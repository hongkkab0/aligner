
from mmrotate.registry import TRANSFORMS
from mmrotate.datasets.transforms import RandomRotate
from aligner_engine.mm_rotate_det.dice.aug_util import get_bounding_rect_of_rboxes
import numpy as np
import math
from aligner_engine.mm_rotate_det.dice.dice_shift_to_center import DiceShiftToCenter


@TRANSFORMS.register_module()
class DiceRandomRotate(RandomRotate):
    DEGREE_MARGIN = 1  # degree
    MAX_DEGREE = 180
    diceShiftToCenter = DiceShiftToCenter(prob=1.0)

    def _dice_random_angle(self, angle):
        return angle * (2 * np.random.rand() - 1)


    def transform(self, results: dict) -> dict:
        """The transform function."""
        if not self._is_rotate():
            return results

        results = self.diceShiftToCenter(results)

        img_shape = results['img'].shape[:2]
        img_width = img_shape[1]
        img_height = img_shape[0]
        img_short_side = min(img_width, img_height)

        center_img = np.array([img_width/2, img_height/2])

        upper_left_boxes, lower_right_boxes = get_bounding_rect_of_rboxes(results['gt_bboxes'], img_width, img_height)
        left_boxes = upper_left_boxes[0]
        right_boxes = lower_right_boxes[0]
        top_boxes = upper_left_boxes[1]
        bot_boxes = lower_right_boxes[1]

        upper_right_boxes = [right_boxes, top_boxes]
        lower_left_boxes = [left_boxes, bot_boxes]
        corners = np.array([upper_left_boxes, upper_right_boxes, lower_right_boxes, lower_left_boxes])
        most_distant_point_from_center = np.abs(corners - center_img).max(axis=0)
        most_distant_x = most_distant_point_from_center[0]
        most_distant_y = most_distant_point_from_center[1]

        width_virtual_box = most_distant_x * 2
        height_virtual_box = most_distant_y * 2
        diagonal_virtual_box = math.sqrt(width_virtual_box**2 + height_virtual_box**2)

        instant_angle_range = 0
        if diagonal_virtual_box <= img_short_side:
            instant_angle_range = self.MAX_DEGREE
        else:
            diff_height = abs(img_height - height_virtual_box)
            diff_width = abs(img_width - width_virtual_box)
            short_diff = min(diff_height, diff_width)

            instant_angle_range = max(0.0, math.asin(short_diff / diagonal_virtual_box) * 180.0 / math.pi - self.DEGREE_MARGIN)

        rotate_angle = self._dice_random_angle(instant_angle_range)
        if self.rect_obj_labels is not None and 'gt_bboxes_labels' in results:
            for label in self.rect_obj_labels:
                if (results['gt_bboxes_labels'] == label).any():
                    rotate_angle = self._random_horizontal_angle()
                    break

        self.rotate.rotate_angle = rotate_angle
        return self.rotate(results)

