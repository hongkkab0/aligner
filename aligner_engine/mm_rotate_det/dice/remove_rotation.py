import cv2
import numpy as np


def remove_rotation(qbox):
    qbox = np.array(qbox, dtype=int)
    qbox = qbox.reshape(4, 2)
    rect = cv2.minAreaRect(qbox)
    if rect[2] >= 45.0:
        points = cv2.boxPoints((rect[0], rect[1], 90.0))
    else:
        points = cv2.boxPoints((rect[0], rect[1], 0.0))

    qbox_flat = [points[0][0], points[0][1],
                 points[1][0], points[1][1],
                 points[2][0], points[2][1],
                 points[3][0], points[3][1]]
    return qbox_flat