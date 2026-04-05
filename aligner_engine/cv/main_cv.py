import sys
import os
import numpy as np
import cv2
import math


def main():
    tmpl = cv2.imread(r"D:\TmpData\images\ref.bmp", cv2.IMREAD_GRAYSCALE)
    tmpl_resize = cv2.resize(tmpl, (0, 0), None, 0.25, 0.25)

    rotated_image_name = r"D:\TmpData\images\UpperFront_20240111204238.bmp"
    image_name = r"D:\TmpData\images\UpperFront_20240111165938.bmp"

    src = cv2.imread(rotated_image_name, cv2.IMREAD_GRAYSCALE)
    src_resize = cv2.resize(src, (0, 0), None, 0.25, 0.25)

    # (y, x)
    pts = [(282, 257), (282, 834), (565, 834), (565, 257)]
    pts_r = [(368, 290), (278, 860), (546, 906), (638, 334)]

    # (x, y)
    pts_r = [[862, 275], [290, 368], [906, 546], [333, 638]]
    pts1 = np.array(pts_r)
    fine_search(tmpl_resize, src_resize, pts1, 32, True)


def fine_search(template, source, pts, padding = 32, show_result = False):
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    # math atan2() and rotate_image() has opposite direction. so should change sign.
    angle = -1.0 * math.degrees(math.atan2(box[1][1] - box[0][1], box[1][0] - box[0][0]))
    center = np.intp(rect[0])

    (height, width) = template.shape[:2]
    pad_x, pad_y = calculate_rotating_padding(width, height, angle)

    src_crop = crop_image(source, center, width+pad_x, height+pad_y, padding)
    mask = np.ones_like(template) * 255

    if abs(angle) > 1.0:
        template = rotate_bound(template, angle)
        mask = rotate_bound(mask, angle)

    tmpl_blur = cv2.blur(template, (3, 3))
    src_blur = cv2.blur(src_crop, (3, 3))

    matchloc = find_template(tmpl_blur, src_blur, mask)

    if abs(angle) > 1.0:
        pts = adjust_rotated_matchloc(matchloc, tmpl_blur.shape[1], tmpl_blur.shape[0], pad_x, pad_y, angle)
    else:
        pts = calculate_match_rect(matchloc, tmpl_blur)

    if show_result:
        display_result(pts, src_crop)

    return pts


def find_template(template, source, mask=None):
    # TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED
    method = cv2.TM_CCORR
    result = cv2.matchTemplate(source, template, method)

    minv, maxv, minloc, maxloc = cv2.minMaxLoc(result)

    if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
        matchloc = minloc
    else:
        matchloc = maxloc

    return matchloc


def crop_image(image, center, width, height, padding):
    w1 = width // 2
    h1 = height // 2

    lefttop = (center[0] - w1 - padding, center[1] - h1 - padding)
    rightbottom = (center[0] + w1 + padding, center[1] + h1 + padding)

    return image[lefttop[1]: rightbottom[1], lefttop[0]: rightbottom[0]]


def rotate_bound(image, angle):
    (height, width) = image.shape[:2]

    pad_x, pad_y = calculate_rotating_padding(width, height, angle)
    padded_img = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    (height, width) = padded_img.shape[:2]
    (cX, cY) = (width // 2, height // 2)

    r_mat = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated_img = cv2.warpAffine(padded_img, r_mat, (width, height))

    return rotated_img


def calculate_rotating_padding(width, height, angle):
    if abs(angle) < 1.0:
        return 0, 0

    l = ((width ** 2 + height ** 2) ** (1 / 2)) / 2.0
    theta = math.atan2(height, width)
    angle_rad = (math.pi * (angle / 180.0))

    if angle > 0:
        theta_y = theta + angle_rad
        theta_x = theta - angle_rad
    else:
        theta_y = theta - angle_rad
        theta_x = theta + angle_rad

    pad_x = int(abs(l * (math.cos(theta_x) - math.cos(theta))))
    pad_y = int(abs(l * (math.sin(theta_y) - math.sin(theta))))

    return pad_x, pad_y


def calculate_match_rect(matchloc, template):
    height, width = template.shape[:2]
    lt = (matchloc[0], matchloc[1])
    rt = (matchloc[0] + width, matchloc[1])
    rb = (matchloc[0] + width, matchloc[1] + height)
    lb = (matchloc[0], matchloc[1] + height)

    return [lt, rt, rb, lb]


def adjust_rotated_matchloc(matchloc, width, height, pad_x, pad_y, angle):
    offset_x = pad_x * 2
    offset_y = pad_y * 2

    if angle > 0.0:
        lt = (matchloc[0], matchloc[1] + offset_y)
        rt = (matchloc[0] + width - offset_x, matchloc[1])
        rb = (matchloc[0] + width, matchloc[1] + height - offset_y)
        lb = (matchloc[0] + offset_x, matchloc[1] + height)
        pts = [lt, rt, rb, lb]
    else:
        lt = (matchloc[0] + offset_x, matchloc[1])
        rt = (matchloc[0] + width, matchloc[1] + offset_y)
        rb = (matchloc[0] + width - offset_x, matchloc[1] + height)
        lb = (matchloc[0], matchloc[1] + height - offset_y)
        pts = [lt, rt, rb, lb]

    return pts


def show_image(image, caption=''):
    if not caption:
        caption = 'result'

    cv2.imshow(caption, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def display_result(pts, source):
    points = np.array(pts).reshape(1, -1, 2)
    dst = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    cv2.polylines(dst, points, isClosed=True, color=(0, 255, 0), thickness=2)

    show_image(dst, 'result')


if __name__ == '__main__':
    main()


def call_test_angle():
    angles = [-90, -60, -30, 0, 30, 60, 90]

    for angle in angles:
        test_angle(angle)


def test_angle(theta):
    image = np.zeros((512, 512, 3), np.uint8)
    lt = [100, 100]
    rt = [300, 100]
    rb = [300, 200]
    lb = [100, 200]

    pts = np.array([lt, rt, rb, lb])
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 1)

    r_lt = rotate_point(lt, theta)
    r_rt = rotate_point(rt, theta)
    r_rb = rotate_point(rb, theta)
    r_lb = rotate_point(lb, theta)

    r_pts = np.array([r_lt, r_rt, r_rb, r_lb])
    r_rect = cv2.minAreaRect(r_pts)
    r_box = cv2.boxPoints(r_rect)
    r_box = np.intp(r_box)
    cv2.drawContours(image, [r_box], 0, (0, 0, 255), 1)

    center, (w, h), angle = cv2.minAreaRect(r_pts)

    print('width: ', round(w), 'height: ', round(h), 'original angle: ', theta, '-> rotated angle:', round(angle))

    return angle


    #show_image(image, 'rect')


def rotate_point(pt, theta):
    x = pt[0]
    y = pt[1]

    angle_rad = (math.pi * (theta / 180.0))
    new_x = x*math.cos(angle_rad) - y*math.sin(angle_rad)
    new_y = x*math.sin(angle_rad) + y*math.cos(angle_rad)

    return [int(new_x), int(new_y)]