import copy
import os
import os.path as osp
import re
import tempfile
import zipfile
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

from mmengine.logging import MMLogger

from mmrotate.evaluation import eval_rbbox_map
from mmcv.ops import box_iou_rotated
from mmrotate.registry import METRICS

from mmrotate.evaluation.metrics import DOTAMetric
import numpy as np
import math
import torch

from aligner_engine.mm_rotate_det.dice.DICErbox2qbox import dice_rbox_to_qbox_single_with_sort_rule


@METRICS.register_module()
class DiceDOTAMetric(DOTAMetric):
    """DOTA evaluation metric.

    Note:  In addition to format the output results to JSON like CocoMetric,
    it can also generate the full image's results by merging patches' results.
    The premise is that you must use the tool provided by us to crop the DOTA
    large images, which can be found at: ``tools/data/dota/split``.

    Args:
        iou_thrs (float or List[float]): IoU threshold. Defaults to 0.5.
        scale_ranges (List[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None.
        metric (str | list[str]): Metrics to be evaluated. Only support
            'mAP' now. If is list, the first setting in the list will
             be used to evaluate metric.
        predict_box_type (str): Box type of model results. If the QuadriBoxes
            is used, you need to specify 'qbox'. Defaults to 'rbox'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format. Defaults to False.
        outfile_prefix (str, optional): The prefix of json/zip files. It
            includes the file path and the prefix of filename, e.g.,
            "a/b/prefix". If not specified, a temp file will be created.
            Defaults to None.
        merge_patches (bool): Generate the full image's results by merging
            patches' results.
        iou_thr (float): IoU threshold of ``nms_rotated`` used in merge
            patches. Defaults to 0.1.
        eval_mode (str): 'area' or '11points', 'area' means calculating the
            area under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1].
            The PASCAL VOC2007 defaults to use '11points', while PASCAL
            VOC2012 defaults to use 'area'. Defaults to '11points'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    @staticmethod
    def _safe_mean(values):
        if not values:
            return np.nan
        return float(np.asarray(values).mean())

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()
        if self.merge_patches:
            # convert predictions to txt format and dump to zip file
            zip_path = self.merge_results(preds, outfile_prefix)
            logger.info(f'The submission file save at {zip_path}')
            return eval_results
        else:
            # convert predictions to coco format and dump to json file
            _ = self.results2json(preds, outfile_prefix)
            if self.format_only:
                logger.info('results are saved in '
                            f'{osp.dirname(outfile_prefix)}')
                return eval_results

        pred_bboxes = [pred['bboxes'] for pred in preds]
        pred_labels = [pred['labels'] for pred in preds]

        gt_bboxes = [gt['bboxes'] for gt in gts]
        gt_labels = [gt['labels'] for gt in gts]

        results = dict()
        results['corner_error'] = []
        results['corner_dx'] = []
        results['corner_dy'] = []
        results['center_error'] = []
        results['center_dx'] = []
        results['center_dy'] = []
        results['longside'] = []
        results['shortside'] = []
        per_class_results = defaultdict(lambda: {
            'corner_error': [],
            'corner_dx': [],
            'corner_dy': [],
            'center_error': [],
            'center_dx': [],
            'center_dy': [],
            'longside': [],
            'shortside': [],
        })

        for (img_gt_labels, img_gt_bboxes, img_pred_labels, img_pred_bboxes) in zip(gt_labels, gt_bboxes, pred_labels,
                                                                                    pred_bboxes):
            matched_indices = set()
            for i, (gt_label, gt_bbox) in enumerate(zip(img_gt_labels, img_gt_bboxes)):
                iou_thr = 0.3
                best_match_idx = None
                if len(img_pred_labels) == 0:
                    continue
                for j, (pred_label, pred_bbox) in enumerate(zip(img_pred_labels, img_pred_bboxes)):
                    if pred_label == gt_label and j not in matched_indices:
                        iou = box_iou_rotated(torch.from_numpy(pred_bbox).unsqueeze(0),
                                              torch.from_numpy(gt_bbox).unsqueeze(0))[0][0]
                        if iou > iou_thr:
                            best_match_idx = j
                            break

                gt_qbbox, gt_sort_rule = dice_rbox_to_qbox_single_with_sort_rule(torch.asarray(np.array(gt_bbox)))

                if best_match_idx is not None:
                    matched_indices.add(best_match_idx)
                    pred_qbbox, _ = dice_rbox_to_qbox_single_with_sort_rule(torch.asarray(np.asarray(img_pred_bboxes[best_match_idx])), gt_sort_rule)

                    diffs = (gt_qbbox - pred_qbbox)
                    diffs_reshape = diffs.reshape((4, 2))
                    corner_dx = abs(diffs_reshape[:, 0]).mean()
                    corner_dy = abs(diffs_reshape[:, 1]).mean()
                    corner_error = ((diffs_reshape ** 2).sum(axis=1) ** 0.5).mean()

                    pred_bbox=img_pred_bboxes[best_match_idx]
                    center_dx = abs(gt_bbox[0] - pred_bbox[0])
                    center_dy = abs(gt_bbox[1] - pred_bbox[1])
                    center_error = (center_dx ** 2 + center_dy ** 2) ** 0.5

                    ls_error = abs(gt_bbox[2:4].max() - pred_bbox[2:4].max())
                    ss_error = abs(gt_bbox[2:4].min() - pred_bbox[2:4].min())
                else:
                    diffs = gt_qbbox
                    diffs_reshape = diffs.reshape((4, 2))
                    corner_dx = abs(diffs_reshape[:, 0]).mean()
                    corner_dy = abs(diffs_reshape[:, 1]).mean()
                    corner_error = ((diffs_reshape ** 2).sum(axis=1) ** 0.5).mean()

                    center_dx = gt_bbox[0]
                    center_dy = gt_bbox[1]
                    center_error = (center_dx ** 2 + center_dy ** 2) ** 0.5
                    ls_error = gt_bbox[2:4].max()
                    ss_error = gt_bbox[2:4].min()

                results['corner_error'].append(corner_error)
                results['corner_dx'].append(corner_dx)
                results['corner_dy'].append(corner_dy)
                results['center_error'].append(center_error)
                results['center_dx'].append(center_dx)
                results['center_dy'].append(center_dy)
                results['longside'].append(ls_error)
                results['shortside'].append(ss_error)
                per_class_results[int(gt_label)]['corner_error'].append(corner_error)
                per_class_results[int(gt_label)]['corner_dx'].append(corner_dx)
                per_class_results[int(gt_label)]['corner_dy'].append(corner_dy)
                per_class_results[int(gt_label)]['center_error'].append(center_error)
                per_class_results[int(gt_label)]['center_dx'].append(center_dx)
                per_class_results[int(gt_label)]['center_dy'].append(center_dy)
                per_class_results[int(gt_label)]['longside'].append(ls_error)
                per_class_results[int(gt_label)]['shortside'].append(ss_error)

        eval_results['mPE'] = {'corner_error': self._safe_mean(results['corner_error']),
                               'corner_dx': self._safe_mean(results['corner_dx']),
                               'corner_dy': self._safe_mean(results['corner_dy']),
                               'center_error': self._safe_mean(results['center_error']),
                               'center_dx': self._safe_mean(results['center_dx']),
                               'center_dy': self._safe_mean(results['center_dy']),
                               'longside': self._safe_mean(results['longside']),
                               'shortside': self._safe_mean(results['shortside'])}
        eval_results['mPE_by_class'] = {}
        for class_idx, class_results in per_class_results.items():
            eval_results['mPE_by_class'][class_idx] = {
                'corner_error': self._safe_mean(class_results['corner_error']),
                'corner_dx': self._safe_mean(class_results['corner_dx']),
                'corner_dy': self._safe_mean(class_results['corner_dy']),
                'center_error': self._safe_mean(class_results['center_error']),
                'center_dx': self._safe_mean(class_results['center_dx']),
                'center_dy': self._safe_mean(class_results['center_dy']),
                'longside': self._safe_mean(class_results['longside']),
                'shortside': self._safe_mean(class_results['shortside']),
            }

        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            dataset_name = self.dataset_meta['classes']
            dets = [pred['pred_bbox_scores'] for pred in preds]

            mean_aps = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, eval_detail = eval_rbbox_map(
                    dets,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=self.use_07_metric,
                    box_type=self.predict_box_type,
                    dataset=dataset_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                eval_results[f'AP{int(iou_thr * 100):02d}_detail'] = eval_detail
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        else:
            raise NotImplementedError
        return eval_results
