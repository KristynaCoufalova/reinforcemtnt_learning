#!/usr/bin/env python3
import argparse
from math import log
from typing import Callable
import unittest

import torch

TOP: int = 0
LEFT: int = 1
BOTTOM: int = 2
RIGHT: int = 3

def bboxes_area(bboxes: torch.Tensor) -> torch.Tensor:
    return torch.relu(bboxes[..., BOTTOM] - bboxes[..., TOP]) \
        * torch.relu(bboxes[..., RIGHT] - bboxes[..., LEFT])

def bboxes_iou(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    intersections = torch.stack([
        torch.maximum(xs[..., TOP], ys[..., TOP]),
        torch.maximum(xs[..., LEFT], ys[..., LEFT]),
        torch.minimum(xs[..., BOTTOM], ys[..., BOTTOM]),
        torch.minimum(xs[..., RIGHT], ys[..., RIGHT]),
    ], dim=-1)

    xs_area, ys_area, intersections_area = bboxes_area(xs), bboxes_area(ys), bboxes_area(intersections)

    return intersections_area / (xs_area + ys_area - intersections_area)

def bboxes_to_rcnn(anchors: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
    ay = (anchors[..., TOP] + anchors[..., BOTTOM]) / 2
    ax = (anchors[..., LEFT] + anchors[..., RIGHT]) / 2
    ah = anchors[..., BOTTOM] - anchors[..., TOP]
    aw = anchors[..., RIGHT] - anchors[..., LEFT]

    by = (bboxes[..., TOP] + bboxes[..., BOTTOM]) / 2
    bx = (bboxes[..., LEFT] + bboxes[..., RIGHT]) / 2
    bh = bboxes[..., BOTTOM] - bboxes[..., TOP]
    bw = bboxes[..., RIGHT] - bboxes[..., LEFT]

    return torch.stack([
        (by - ay) / ah,
        (bx - ax) / aw,
        torch.log(bh / ah),
        torch.log(bw / aw)
    ], dim=-1)

def bboxes_from_rcnn(anchors: torch.Tensor, rcnns: torch.Tensor) -> torch.Tensor:
    ay = (anchors[..., TOP] + anchors[..., BOTTOM]) / 2
    ax = (anchors[..., LEFT] + anchors[..., RIGHT]) / 2
    ah = anchors[..., BOTTOM] - anchors[..., TOP]
    aw = anchors[..., RIGHT] - anchors[..., LEFT]

    by = rcnns[..., 0] * ah + ay
    bx = rcnns[..., 1] * aw + ax
    bh = torch.exp(rcnns[..., 2]) * ah
    bw = torch.exp(rcnns[..., 3]) * aw

    top = by - bh / 2
    left = bx - bw / 2
    bottom = by + bh / 2
    right = bx + bw / 2

    return torch.stack([top, left, bottom, right], dim=-1)

def bboxes_training(
    anchors: torch.Tensor, gold_classes: torch.Tensor, gold_bboxes: torch.Tensor, iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_anchors = anchors.shape[0]
    anchor_classes = torch.zeros(num_anchors, dtype=torch.int64)
    anchor_bboxes = torch.zeros((num_anchors, 4), dtype=torch.float32)

    ious = bboxes_iou(anchors[:, None, :], gold_bboxes[None, :, :])  # [anchors, gold]
    best_anchors = torch.argmax(ious, dim=0)

    assigned = torch.zeros(num_anchors, dtype=torch.bool)
    for gold_idx, anchor_idx in enumerate(best_anchors):
        if not assigned[anchor_idx]:
            assigned[anchor_idx] = True
            anchor_classes[anchor_idx] = 1 + gold_classes[gold_idx]
            anchor_bboxes[anchor_idx] = bboxes_to_rcnn(
                anchors[anchor_idx].unsqueeze(0), gold_bboxes[gold_idx].unsqueeze(0)
            )[0]

    for anchor_idx in range(num_anchors):
        if not assigned[anchor_idx]:
            best_gold_iou, best_gold_idx = torch.max(ious[anchor_idx], dim=0)
            if best_gold_iou >= iou_threshold:
                assigned[anchor_idx] = True
                anchor_classes[anchor_idx] = 1 + gold_classes[best_gold_idx]
                anchor_bboxes[anchor_idx] = bboxes_to_rcnn(
                    anchors[anchor_idx].unsqueeze(0), gold_bboxes[best_gold_idx].unsqueeze(0)
                )[0]

    return anchor_classes, anchor_bboxes

def main(args: argparse.Namespace) -> tuple[Callable, Callable, Callable]:
    return bboxes_to_rcnn, bboxes_from_rcnn, bboxes_training

class Tests(unittest.TestCase):
    def test_bboxes_to_from_rcnn(self):
        data = [
            [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]],
            [[0, 0, 10, 10], [5, 0, 15, 10], [.5, 0, 0, 0]],
            [[0, 0, 10, 10], [0, 5, 10, 15], [0, .5, 0, 0]],
            [[0, 0, 10, 10], [0, 0, 20, 30], [.5, 1, log(2), log(3)]],
            [[0, 9, 10, 19], [2, 10, 5, 16], [-0.15, -0.1, -1.20397, -0.51083]],
            [[5, 3, 15, 13], [7, 7, 10, 9], [-0.15, 0, -1.20397, -1.60944]],
            [[7, 6, 17, 16], [9, 10, 12, 13], [-0.15, 0.05, -1.20397, -1.20397]],
            [[5, 6, 15, 16], [7, 7, 10, 10], [-0.15, -0.25, -1.20397, -1.20397]],
            [[6, 3, 16, 13], [8, 5, 12, 8], [-0.1, -0.15, -0.91629, -1.20397]],
            [[5, 2, 15, 12], [9, 6, 12, 8], [0.05, 0, -1.20397, -1.60944]],
            [[2, 10, 12, 20], [6, 11, 8, 17], [0, -0.1, -1.60944, -0.51083]],
            [[10, 9, 20, 19], [12, 13, 17, 16], [-0.05, 0.05, -0.69315, -1.20397]],
            [[6, 7, 16, 17], [10, 11, 12, 14], [0, 0.05, -1.60944, -1.20397]],
            [[2, 2, 12, 12], [3, 5, 8, 8], [-0.15, -0.05, -0.69315, -1.20397]],
        ]
        for anchors, bboxes, rcnns in [map(lambda x: [x], row) for row in data] + [zip(*data)]:
            anchors, bboxes, rcnns = [torch.tensor(data, dtype=torch.float32) for data in [anchors, bboxes, rcnns]]
            torch.testing.assert_close(bboxes_to_rcnn(anchors, bboxes), rcnns, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(bboxes_from_rcnn(anchors, rcnns), bboxes, atol=1e-3, rtol=1e-3)

    def test_bboxes_training(self):
        anchors = torch.tensor([[0, 0, 10, 10], [0, 10, 10, 20], [10, 0, 20, 10], [10, 10, 20, 20]])
        for gold_classes, gold_bboxes, anchor_classes, anchor_bboxes, iou in [
                [[1], [[14, 14, 16, 16]], [0, 0, 0, 2], [[0, 0, 0, 0]] * 3 + [[0, 0, log(.2), log(.2)]], 0.5],
                [[2], [[0, 0, 20, 20]], [3, 0, 0, 0], [[.5, .5, log(2), log(2)]] + [[0, 0, 0, 0]] * 3, 0.26],
                [[2], [[0, 0, 20, 20]], [3, 3, 3, 3],
                 [[y, x, log(2), log(2)] for y in [.5, -.5] for x in [.5, -.5]], 0.24],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 0, 1],
                 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [-0.35, -0.45, 0.53062, 0.40546]], 0.5],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 2, 1],
                 [[0, 0, 0, 0], [0, 0, 0, 0], [-0.1, 0.6, -0.22314, 0.69314], [-0.35, -0.45, 0.53062, 0.40546]], 0.3],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 1, 2, 1],
                 [[0, 0, 0, 0], [0.65, -0.45, 0.53062, 0.40546], [-0.1, 0.6, -0.22314, 0.69314],
                  [-0.35, -0.45, 0.53062, 0.40546]], 0.17],
        ]:
            gold_classes, anchor_classes = torch.tensor(gold_classes), torch.tensor(anchor_classes)
            gold_bboxes, anchor_bboxes = torch.tensor(gold_bboxes), torch.tensor(anchor_bboxes)
            computed_classes, computed_bboxes = bboxes_training(anchors, gold_classes, gold_bboxes, iou)
            torch.testing.assert_close(computed_classes, anchor_classes, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(computed_bboxes, anchor_bboxes, atol=1e-3, rtol=1e-3)

if __name__ == '__main__':
    unittest.main()
