from mmrotate.structures import RotatedBoxes
import numpy as np

def get_bounding_rect_of_rboxes(rboxes: RotatedBoxes, img_width, img_height):
    points = []  # [[x,y],[x,y],...]
    for rbox in rboxes.tensor:
        qbox = RotatedBoxes.rbox2corner(rbox.clone().detach()).tolist()
        points.extend(qbox)

    points = np.array(points)
    if len(points) == 0:
        upper_left_of_all_boxes = np.array([img_width/2, img_height/2])
        lower_right_of_all_boxes = np.array([img_width/2, img_height/2])
    else:
        upper_left_of_all_boxes = points.min(axis=0)
        lower_right_of_all_boxes = points.max(axis=0)

    return upper_left_of_all_boxes, lower_right_of_all_boxes



