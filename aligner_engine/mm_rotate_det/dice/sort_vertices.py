import math
import torch

MAX_NUM_VERT_IDX = 9
INTERSECTION_OFFSET = 8
EPSILON = 1e-8


def compare_vertices(x1: float, y1: float, x2: float, y2: float):
    if math.fabs(x1 - x2) < EPSILON and math.fabs(y2 - y1) < EPSILON:
        return False

    if y1 > 0 > y2:
        return True
    if y2 > 0 > y1:
        return False

    n1 = x1 * x1 + y1 * y1 + EPSILON
    n2 = x2 * x2 + y2 * y2 + EPSILON
    diff = math.fabs(x1) * x1 / n1 - math.fabs(x2) * x2 / n2

    if y1 > 0 and y2 > 0:
        if diff > EPSILON:
            return True
        else:
            return False

    if y1 < 0 and y2 < 0:
        if diff < EPSILON:
            return True
        else:
            return False

    return False


def diff_iou_rotated_sort_vertices_forward_cpu(vertices, mask, num_valid):
    b = vertices.size(0)
    n = vertices.size(1)
    m = vertices.size(2)

    idx = torch.zeros((b, n, MAX_NUM_VERT_IDX), dtype=torch.int32).to(vertices.device)

    for batch_idx in range(b):
        for i in range(n):
            pad = 0  # 임의의 잘못된 교차점의 인덱스
            for j in range(INTERSECTION_OFFSET, m):
                if not mask[batch_idx, i, j]:
                    pad = j
                    break

            if num_valid[batch_idx, i] < 3:
                # 유효한 꼭짓점이 충분하지 않으면, 잘못된 교차점을 사용합니다.
                idx[batch_idx, i, :] = pad
            else:
                for j in range(num_valid[batch_idx, i]):
                    x_min = 1
                    y_min = -EPSILON
                    i_take = 0

                    if j != 0:
                        i2 = idx[batch_idx, i, j - 1]
                        x2 = vertices[batch_idx, i, i2, 0]
                        y2 = vertices[batch_idx, i, i2, 1]

                    for k in range(m):
                        x = vertices[batch_idx, i, k, 0]
                        y = vertices[batch_idx, i, k, 1]

                        if mask[batch_idx, i, k] and compare_vertices(x, y, x_min, y_min):
                            if j == 0 or (j != 0 and compare_vertices(x2, y2, x, y)):
                                x_min = x
                                y_min = y
                                i_take = k

                    idx[batch_idx, i, j] = i_take

                if num_valid[batch_idx, i] > 8:
                    idx[batch_idx, i, 8] = idx[batch_idx, i, 0]
                    idx[batch_idx, i, 9:] = pad
                else:
                    # 첫 번째 인덱스 복제
                    idx[batch_idx, i, num_valid[batch_idx, i]] = idx[batch_idx, i, 0]
                    # 패딩
                    idx[batch_idx, i, num_valid[batch_idx, i] + 1:] = pad

                # 두 박스가 정확히 동일한 경우 처리
                if num_valid[batch_idx, i] == 8:
                    counter = 0
                    for j in range(4):
                        check = idx[batch_idx, i, j]
                        for k in range(4, INTERSECTION_OFFSET):
                            if idx[batch_idx, i, k] == check:
                                counter += 1

                    if counter == 4:
                        idx[batch_idx, i, 4] = idx[batch_idx, i, 0]
                        idx[batch_idx, i, 5:] = pad

    return idx
