# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.dataset import BaseDataset
from aligner_engine.mm_rotate_det.dice.remove_rotation import remove_rotation

from mmrotate.registry import DATASETS
import aligner_engine.utils as util
import os



@DATASETS.register_module()
class DiceRotateDetDataset(BaseDataset):
    """DOTA-v1.0 dataset for detection.

    Note: ``ann_file`` in DOTADataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTADataset,
    it is the path of a folder containing XML files.
    
    Args:
        diff_thr (int): The difficulty threshold of ground truth. Bboxes
            with difficulty higher than it will be ignored. The range of this
            value should be non-negative integer. Defaults to 100.
        img_suffix (str): The suffix of images. Defaults to 'png'.
    """

    METAINFO = {
        'classes':
            ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
             'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
             'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
             'harbor', 'swimming-pool', 'helicopter'),

        # palette is a list of color tuples, which is used for visualization.
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                    (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                    (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
                    (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
                    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
                    (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
                    (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
                    (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
                    (134, 134, 103), (145, 148, 174), (255, 208, 186),
                    (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
                    (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
                    (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
                    (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
                    (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
                    (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
                    (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
                    (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
                    (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
                    (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
                    (191, 162, 208)]
    }

    MINIMUM_DATA_SUMMARY = 400

    def __init__(self,
                 **kwargs) -> None:

        self.summary_file = kwargs['ann_file']
        self.class_meta = self._load_summary()
        classes = [cls_dict["name"] for cls_dict in self.class_meta]
        kwargs['metainfo'] = {'classes': classes}

        self.no_rotation = kwargs['no_rotation']
        del kwargs['no_rotation']

        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based

        data_list = []
        summary_dict = util.read_json(self.summary_file)
        len_data_summary = len(summary_dict['data_summary'])

        repeat_data = 1

        if self.test_mode == False:
            if len_data_summary < self.MINIMUM_DATA_SUMMARY:
                repeat_data = int(self.MINIMUM_DATA_SUMMARY / len_data_summary)
                if repeat_data == 0:
                    repeat_data = 1

        img_idx = 0
        for data in summary_dict['data_summary']:
            label_path = data['label_path']
            label = util.read_json(label_path)

            for repeate_idx in range(repeat_data):
                data_info = {}
                data_info['img_id'] = str(img_idx)
                data_info['img_path'] = data['img_path']
                data_info['file_name'] = os.path.basename(data['img_path'])

                instances = []
                for shape in label['shapes']:
                    instance = {}
                    instance['bbox'] = [shape['x1'], shape['y1'],
                                        shape['x2'], shape['y2'],
                                        shape['x3'], shape['y3'],
                                        shape['x4'], shape['y4']]
                    if self.no_rotation:
                        instance['bbox'] = remove_rotation(instance['bbox'])

                    class_name = shape['label']
                    instance['bbox_label'] = cls_map[class_name]
                    instance['ignore_flag'] = 0
                    instances.append(instance)

                data_info['instances'] = instances
                data_list.append(data_info)
                img_idx = img_idx + 1

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DOTA category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]

    def _load_summary(self):
        try:
            summary_dict = util.read_json(self.summary_file)
            if not summary_dict['task_type'] == 'rotate_det':
                raise OSError(
                    'task_type: det is required for building DICE Rotated detection dataset')
            return summary_dict['class_summary']['classes']
        except Exception as e:
            raise type(e)(f'{self.summary_file}: {e}')
