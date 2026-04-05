from __future__ import annotations

import aligner_engine.utils as util
from aligner_engine.default_project_config import get_default_project_config
from aligner_engine.model_profiles import get_model_profile, list_model_profiles, resolve_pretrained_checkpoint
from aligner_engine.project_settings import ProjectSettings
import aligner_engine.const as const
from aligner_engine.checkpoint_store import ProjectCheckpointStore
from copy import deepcopy
import json
import logging
import traceback
import warnings
from aligner_engine.summary import TrainSummary, ResultSummary
from aligner_engine.best_ckpt_measure import BestCkptMeasure
import os
import time
import numpy as np
from shutil import copyfile
import shutil

APP_LOGGER = logging.getLogger("aligner.app")
TRAIN_LOGGER = logging.getLogger("aligner.trainer")
TEST_LOGGER = logging.getLogger("aligner.tester")


class Worker:

    def __init__(self, project_path, is_new: bool):
        self._project_path = project_path
        self._is_new = is_new
        self._project_config = {}
        self._project_settings = ProjectSettings.from_dict(
            get_default_project_config(),
            default_model_profile=get_model_profile(None).id,
        )

        self._runner = None  # for training
        self._stop_loop_hook = None  # for training

        self._detector = None  # for test
        self._stop_test: bool = False  # for test
        self.mean_test_time = -1.0  # for test
        self._test_artifact_paths = []

        self._train_summary: TrainSummary = TrainSummary()
        self._train_result_summary: ResultSummary = ResultSummary()
        self._valid_result_summary: ResultSummary = ResultSummary()
        self._test_result_summary: ResultSummary = ResultSummary()
        self._best_ckpt_measure: BestCkptMeasure = BestCkptMeasure()
        self._register_init_finished: bool = False
        self.metric_name = "mAP"

        self._project_config = self._project_settings.to_dict()
        if is_new == True:
            self._create_new_project_config()
            self._save_empty_records()  # make empty summary pkl
            self._clear_auto_saved()
            self._clear_data_summary()

        self._load_project_config()
        self._load_records()

    def _create_new_project_config(self):
        project_config_path = util.join_path(self._project_path, const.PROJECT_CONFIG_NAME)
        util.write_yaml(project_config_path, self._project_config)

    def _load_project_config(self):
        project_config = util.read_yaml(util.join_path(self._project_path, const.PROJECT_CONFIG_NAME))
        merged = get_default_project_config()
        if project_config is not None:
            merged.update(project_config)
        self._project_settings = ProjectSettings.from_dict(
            merged,
            default_model_profile=get_model_profile(None).id,
        )
        self._project_config = self._project_settings.to_dict()

    def set_project_config(self, project_config):
        settings = ProjectSettings.from_dict(
            project_config,
            default_model_profile=get_model_profile(None).id,
        )
        self.set_project_settings(settings)

    def set_project_settings(self, settings: ProjectSettings):
        project_config_path = util.join_path(self._project_path, const.PROJECT_CONFIG_NAME)
        util.write_yaml(project_config_path, settings.to_dict())
        self._project_settings = deepcopy(settings)
        self._project_config = self._project_settings.to_dict()

    def get_project_config(self):
        return deepcopy(self._project_config)

    def get_project_settings(self) -> ProjectSettings:
        return deepcopy(self._project_settings)

    def get_model_profiles(self):
        return list_model_profiles()

    def get_training_split_paths(self):
        train_summary_path = util.join_path(self._project_path, "dataset_summary_training.json")
        valid_summary_path = util.join_path(self._project_path, "dataset_summary_validation.json")
        return train_summary_path, valid_summary_path

    def get_last_checkpoint_path(self):
        return util.join_path(self._project_path, const.DIRNAME_AUTOSAVED, const.FILENAME_CKPT_LAST)

    def get_last_completed_epoch(self):
        if len(self._train_summary.tr_by_epoch) <= 0:
            return 0
        return max(self._train_summary.tr_by_epoch.keys())

    def can_resume_training(self):
        return os.path.exists(self.get_last_checkpoint_path())

    def get_train_summary(self) -> TrainSummary:
        return deepcopy(self._train_summary)

    def get_train_result_summary(self) -> ResultSummary:
        return deepcopy(self._train_result_summary)

    def get_valid_result_summary(self) -> ResultSummary:
        return deepcopy(self._valid_result_summary)

    def get_test_result_summary(self) -> ResultSummary:
        return deepcopy(self._test_result_summary)

    @staticmethod
    def _get_gt_from_label(label_path, cls_map, no_rotation: bool):
        import cv2
        import torch
        from aligner_engine.mm_rotate_det.dice.DICErbox2qbox import dice_rbox_to_qbox_single_with_sort_rule

        label = util.read_json(label_path)
        gt_qbboxes, gt_labels, gt_sort_rules = [], [], []
        for shape in label['shapes']:
            (x, y), (w, h), angle = cv2.minAreaRect(
                np.array([[shape['x1'], shape['y1']],
                          [shape['x2'], shape['y2']],
                          [shape['x3'], shape['y3']],
                          [shape['x4'], shape['y4']]], dtype=np.int32))
            if no_rotation:
                if angle >= 45.0:
                    angle = 90.0
                else:
                    angle = 0.0

            gt_rbbox = torch.tensor([x, y, w, h, angle / 180 * np.pi])
            gt_qbbox, gt_sort_rule = dice_rbox_to_qbox_single_with_sort_rule(gt_rbbox)
            gt_qbbox = np.array(gt_qbbox)
            class_name = shape['label']
            gt_label = -1
            if class_name in cls_map:
                gt_label = cls_map[class_name]
            gt_qbboxes.append(gt_qbbox)
            gt_labels.append(gt_label)
            gt_sort_rules.append(gt_sort_rule)
        return gt_qbboxes, gt_labels, gt_sort_rules

    def _random_divide(self, cfg, dataset_summary):
        valid_ratio = cfg.trainval_split.valid_ratio
        keep_split = cfg.trainval_split.keep_split

        data_thr = cfg.trainval_split.data_thr
        max_val = cfg.trainval_split.max_val

        total_cnt = len(dataset_summary['data_summary'])

        trains = dataset_summary.copy()
        tests = dataset_summary.copy()

        train_summary_path = util.join_path(self._project_path, "dataset_summary_training.json")
        test_summary_path = util.join_path(self._project_path, "dataset_summary_validation.json")

        val_cnt = min(int(total_cnt * valid_ratio), max_val, total_cnt)

        if total_cnt >= data_thr or val_cnt == max_val:
            test_data_summary, train_data_summary = self._permutate(total_cnt, val_cnt, dataset_summary)
            trains['data_summary'] = train_data_summary
            tests['data_summary'] = test_data_summary

            TRAIN_LOGGER.info("Dataset was split into a Training set with {} samples, Validation set with {} samples"
                         .format(total_cnt - val_cnt, val_cnt))

        with open(train_summary_path, "w", encoding="utf-8") as f:
            json.dump(trains, f, ensure_ascii=False, indent=4)

        with open(test_summary_path, "w", encoding="utf-8") as f:
            json.dump(tests, f, ensure_ascii=False, indent=4)

        return train_summary_path, test_summary_path

    def _permutate(self, total_cnt, val_cnt, dataset_summary):
        permuted = np.random.permutation(total_cnt)
        test_data_summary, train_data_summary = [], []
        test_data_summary += [dataset_summary['data_summary'][i] for i in permuted[:val_cnt]]
        train_data_summary += [dataset_summary['data_summary'][i] for i in range(total_cnt) if
                               i not in permuted[:val_cnt]]

        is_valid_test_data_summary = False
        for test_data in test_data_summary:
            label_path = test_data['label_path']
            label = util.read_json(label_path)
            if len(label['shapes']) > 0 :
                is_valid_test_data_summary = True
                break

        if is_valid_test_data_summary == False:  ##if valid dataset don't have box
            for train_data in train_data_summary:
                label_path = train_data['label_path']
                label = util.read_json(label_path)
                if len(label['shapes']) > 0:  ##swap
                    temp = test_data_summary[0]
                    test_data_summary[0] = train_data
                    train_data_summary[0] = temp
                    break

        return test_data_summary, train_data_summary


    def start_train(self, callback_one_epoch_finished, callback_one_iter_finished, callback_status=None, resume=False):
        import mmcv
        import torch
        from mmdet.utils import register_all_modules as register_all_modules_mmdet
        from mmrotate.utils import register_all_modules
        from mmengine.config import Config
        from aligner_engine.hooks.dice_stop_loop_hook import DICEStopLoopHook
        from aligner_engine.hooks.dice_gui_callback_hook import DICECallbackHook
        from aligner_engine.mm_rotate_det.dice.dice_rotate_det_runner import DiceRotateDetRunner

        settings = self._project_settings
        dataset_summary_path = self.get_dataset_summary_path()
        emit_status = callback_status if callback_status is not None else lambda _message: None
        resume = bool(resume and self.can_resume_training())
        warnings.filterwarnings(
            "ignore",
            message=r"The `clip` function does nothing in `RotatedBoxes`\.",
            category=UserWarning,
        )

        if resume:
            self._load_records()
            emit_status(f"Resuming training from epoch {self.get_last_completed_epoch() + 1}...")
        else:
            # reset summary
            self._train_summary = TrainSummary()
            self._train_result_summary = ResultSummary()
            self._valid_result_summary = ResultSummary()
            self._best_ckpt_measure = BestCkptMeasure()

        # initialize configs
        emit_status("Loading training config...")
        profile = get_model_profile(settings.model_profile)
        base_config_path = profile.train_config_path(util.ROOT_PATH)
        cfg = Config.fromfile(base_config_path)

        cfg.launcher = "none"
        cfg.work_dir = self._project_path
        cfg.resume = resume
        cfg.load_from = self.get_last_checkpoint_path() if resume else None

        emit_status("Loading dataset summary...")
        with open(dataset_summary_path, 'r', encoding='utf-8') as f:
            dataset_summary = json.load(f)

        class_summary = dataset_summary["class_summary"]
        num_classes = class_summary["num_classes"]

        train_summary_path, test_summary_path = self.get_training_split_paths()
        keep_existing_split = (
            (resume or bool(cfg.trainval_split.keep_split))
            and os.path.exists(train_summary_path)
            and os.path.exists(test_summary_path)
        )
        if keep_existing_split:
            emit_status("Using existing train/validation split...")
        else:
            emit_status("Splitting train/validation dataset...")
            train_summary_path, test_summary_path = self._random_divide(cfg, dataset_summary)

        # for deployment
        with open(train_summary_path, 'r', encoding='utf-8') as f:
            train_dataset_summary = json.load(f)
        sample_image_path = ""
        for sample_data in train_dataset_summary["data_summary"]:
            label_path = sample_data['label_path']
            label = util.read_json(label_path)
            if len(label['shapes']) > 0:
                sample_image_path = sample_data['img_path']
                break

        emit_status("Preparing deployment sample...")
        sample_image = mmcv.imread(sample_image_path)
        mmcv.imwrite(sample_image, os.path.join(self._project_path, const.FILENAME_SAMPLE_IMAGE_FOR_DEPLOYMENT))

        cfg.train_dataloader.dataset.ann_file = train_summary_path
        cfg.val_dataloader.dataset.ann_file = test_summary_path
        cfg.model.bbox_head.num_classes = num_classes
        if not resume:
            pretrained_checkpoint = resolve_pretrained_checkpoint(
                util.ROOT_PATH,
                settings.model_profile,
                settings.model_pretrained_checkpoint,
            )
            if pretrained_checkpoint:
                cfg.model.backbone.init_cfg.checkpoint = pretrained_checkpoint
            else:
                cfg.model.backbone.init_cfg = None
                emit_status("Pretrained checkpoint not found. Training will start from scratch.")
                TRAIN_LOGGER.warning(
                    "Pretrained checkpoint for model profile '%s' was not found. Training will start from scratch.",
                    profile.id,
                )

        cfg.train_cfg.max_epochs = settings.max_epochs
        milestones = [int(settings.max_epochs * 4.0 / 8.0),
                      int(settings.max_epochs * 6.5 / 8.0)]
        param_scheduler = deepcopy(cfg.param_scheduler)
        for scheduler in param_scheduler:
            if scheduler.type == "MultiStepLR":
                scheduler.milestones = milestones

        cfg.param_scheduler = param_scheduler

        train_pipeline = deepcopy(cfg.train_dataloader.dataset.pipeline)
        valid_pipeline = deepcopy(cfg.val_dataloader.dataset.pipeline)

        rescale_size = (settings.resize, settings.resize)
        random_flip_directions = []
        if settings.aug_flip_horizontal_use:
            random_flip_directions.append("horizontal")

        if settings.aug_flip_vertical_use:
            random_flip_directions.append("vertical")

        random_flip_tranform_idx = -1
        for idx, transform in enumerate(train_pipeline):
            if transform.type == "mmdet.Resize":
                transform.scale = rescale_size
            elif transform.type == "mmdet.RandomFlip":
                transform.direction = random_flip_directions
                random_flip_tranform_idx = idx
            elif transform.type == "mmdet.Pad":
                transform.size = rescale_size
            elif transform.type == "mmdet.PackDetInputs":
                if len(random_flip_directions) == 0:
                    transform.meta_keys = (
                        'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')

        if len(random_flip_directions) == 0:
            train_pipeline.pop(random_flip_tranform_idx)

        random_rotate_tranform_idx = -1
        if settings.no_rotation:
            for idx, transform in enumerate(train_pipeline):
                if transform.type == "DiceRandomRotate":
                    random_rotate_tranform_idx = idx
                    break
            train_pipeline.pop(random_rotate_tranform_idx)

        for transform in valid_pipeline:
            if transform.type == "mmdet.Resize":
                transform.scale = rescale_size
            elif transform.type == "mmdet.Pad":
                transform.size = rescale_size

        cfg.no_rotation = settings.no_rotation
        cfg.train_dataloader.dataset.no_rotation = settings.no_rotation
        cfg.val_dataloader.dataset.no_rotation = settings.no_rotation

        cfg.train_dataloader.dataset.pipeline = train_pipeline
        cfg.val_dataloader.dataset.pipeline = valid_pipeline

        # On Windows, dataloader worker spawning can dominate the initial
        # "first batch" delay and make the UI feel unresponsive.
        batch_size = settings.batch_size
        if os.name == "nt":
            train_num_workers = 0
            valid_num_workers = 0
        else:
            train_num_workers = min(max(int(batch_size / 2), 1), 4)
            valid_num_workers = min(train_num_workers, 2)
        cfg.train_dataloader.num_workers = train_num_workers
        cfg.val_dataloader.num_workers = valid_num_workers
        cfg.train_dataloader.persistent_workers = train_num_workers > 0
        cfg.val_dataloader.persistent_workers = valid_num_workers > 0
        cfg.train_dataloader.batch_size = batch_size

        cfg.test_dataloader = cfg.val_dataloader

        classes = []
        for class_info in class_summary['classes']:
            classes.append(class_info['name'])
        cfg.classes_dice = classes
        cfg.test_pipeline = cfg.test_dataloader.dataset.pipeline  # this is for mmdeploy
        base_cfg_path = util.join_path(cfg.work_dir, const.FILENAME_MODEL_CONFIG)
        cfg.dump(base_cfg_path)

        if self._register_init_finished == False:
            emit_status("Registering detection modules...")
            register_all_modules_mmdet(init_default_scope=False)
            register_all_modules(init_default_scope=False)
            import aligner_engine.mm_rotate_det.dice  # to register dice modules

            self._register_init_finished = True

        emit_status("Building training runner...")
        self._runner = DiceRotateDetRunner.from_cfg(cfg)
        self._stop_loop_hook = DICEStopLoopHook()
        self._runner.register_hook(DICECallbackHook(worker=self,
                                                    iter_callback=callback_one_iter_finished,
                                                    epoch_callback=callback_one_epoch_finished))
        self._runner.register_hook(self._stop_loop_hook)

        emit_status("Starting dataloader workers...")
        if not torch.cuda.is_available():
            self._runner.model.to("cpu")
            TRAIN_LOGGER.info("Training started on CPU.")
        else:
            TRAIN_LOGGER.info("Training started on GPU.")
        self._runner.train()

    def stop_training(self):
        self._stop_loop_hook.quit()
        self.close_logger()
        del self._runner
        self._runner = None

    def success_training(self):
        self.close_logger()
        del self._runner
        self._runner = None

    def _cleanup_test_artifacts(self):
        for path in self._test_artifact_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                continue
        self._test_artifact_paths = []

    def _release_detector(self):
        detector = self._detector
        self._detector = None
        if detector is None:
            return
        try:
            detector.to_cpu()
        except Exception:
            pass
        del detector

    def start_test(self, callback_one_iter_finished, img_paths):
        import mmcv
        import torch
        from mmengine.config import Config
        from mmrotate.structures.bbox import qbox2rbox
        from aligner_engine.detector import Detector
        from aligner_engine.mm_rotate_det.dice.DICErbox2qbox import dice_rbox_to_qbox_single_with_sort_rule
        settings = self._project_settings
        warnings.filterwarnings(
            "ignore",
            message=r"The `clip` function does nothing in `RotatedBoxes`\.",
            category=UserWarning,
        )

        try:
            config_path_src = util.join_path(self._project_path, const.DIRNAME_AUTOSAVED,
                                             const.FILENAME_MODEL_CONFIG)
            config_path_dst = util.join_path(self._project_path, "test_" + const.FILENAME_MODEL_CONFIG)
            copyfile(config_path_src, config_path_dst)

            ckpt_path_src = util.join_path(self._project_path, const.DIRNAME_AUTOSAVED,
                                           const.FILENAME_CKPT)
            ckpt_path_dst = util.join_path(self._project_path, "test_" + const.FILENAME_CKPT)
            copyfile(ckpt_path_src, ckpt_path_dst)
            self._test_artifact_paths = [config_path_dst, ckpt_path_dst]
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            error_msg = "ERROR - " + str(e)
            TEST_LOGGER.error(error_msg)
            self._cleanup_test_artifacts()
            return

        machine = "cpu"
        if torch.cuda.is_available():
            machine = "cuda:0"

        deploy_dir = util.join_path(self._project_path, const.DIRNAME_DEPLOYMENT_WORK_DIR)
        deploy_config_path = util.join_path(deploy_dir, const.FILENAME_DEPLOY_CONFIG)
        vino_xml_path = util.join_path(deploy_dir, const.FILENAME_EXPORT_VINO_XML)
        enable_vino = settings.inference_enable_openvino

        self._detector = Detector(
            config_path_dst,
            ckpt_path_dst,
            machine,
            deploy_config_path=deploy_config_path,
            vino_xml_path=vino_xml_path,
            enable_vino=enable_vino,
        )
        cfg = Config.fromfile(config_path_dst)

        self._stop_test = False
        self._test_result_summary: ResultSummary = ResultSummary()
        cls_map = self._train_result_summary.class_index

        total_len = len(img_paths)
        times = []
        try:
            for idx, img_path in enumerate(img_paths):
                img = mmcv.imread(img_path)
                start_time = time.time()
                result = self._detector.inference(img)
                end_time = time.time()
                times.append(end_time - start_time)

                label_path = os.path.splitext(img_path)[0] + ".json"

                if os.path.exists(label_path):
                    preds = [[cls_map[result[key]['class_name']], np.asarray(result[key]['qbox'])] for _, key in
                             enumerate(result)]
                    gt_qbboxes, gt_labels, gt_sort_rules = Worker._get_gt_from_label(label_path, cls_map, cfg.no_rotation)
                    from mmcv.ops import box_iou_quadri
                    for i, (gt_label, gt_qbbox, gt_sort_rule) in enumerate(zip(gt_labels, gt_qbboxes, gt_sort_rules)):
                        matched_indices = set()
                        if (i + 1) > len(preds):
                            break
                        best_match_idx = None
                        for j, [pred_label, pred_qbbox] in enumerate(preds):
                            if pred_label == gt_label and j not in matched_indices:
                                iou = box_iou_quadri(torch.from_numpy(pred_qbbox).float().unsqueeze(0),
                                                     torch.from_numpy(gt_qbbox).float().unsqueeze(0))[0][0]
                                if iou > 0.3:
                                    best_match_idx = j
                                    break

                        gt_bbox = qbox2rbox(torch.from_numpy(gt_qbbox))

                        if gt_label != -1:
                            if best_match_idx is not None:
                                matched_indices.add(best_match_idx)
                                pred_rbox = qbox2rbox(torch.from_numpy(preds[best_match_idx][1]).type(torch.float32))
                                pred_qbox, _ = dice_rbox_to_qbox_single_with_sort_rule(pred_rbox, gt_sort_rule)

                                diffs = (gt_qbbox - np.array(pred_qbox))
                                diffs_reshape = diffs.reshape((4, 2))
                                corner_dx = abs(diffs_reshape[:, 0]).mean()
                                corner_dy = abs(diffs_reshape[:, 1]).mean()
                                corner_error = ((diffs_reshape ** 2).sum(axis=1) ** 0.5).mean()

                                center_dx = abs(gt_bbox[0] - pred_rbox[0])
                                center_dy = abs(gt_bbox[1] - pred_rbox[1])
                                center_error = (center_dx ** 2 + center_dy ** 2) ** 0.5

                                ls_error = abs(gt_bbox[2:4].max() - pred_rbox[2:4].max())
                                ss_error = abs(gt_bbox[2:4].min() - pred_rbox[2:4].min())
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

                            result[str(i)]['corner_error'] = corner_error
                            result[str(i)]['corner_dx'] = corner_dx
                            result[str(i)]['corner_dy'] = corner_dy
                            result[str(i)]['center_error'] = center_error
                            result[str(i)]['center_dx'] = center_dx
                            result[str(i)]['center_dy'] = center_dy
                            result[str(i)]['longside'] = ls_error
                            result[str(i)]['shortside'] = ss_error

                self._test_result_summary.add_data_result(img_path, result)
                callback_one_iter_finished(idx, total_len)

                if self._stop_test:
                    break
        finally:
            self.mean_test_time = np.array(times[-10:]).mean() if len(times) > 0 else -1.0
            self._release_detector()
            self._cleanup_test_artifacts()

    def stop_test(self):
        self._stop_test = True
        self._release_detector()
        self._cleanup_test_artifacts()

    def success_test(self):
        self._release_detector()
        self._cleanup_test_artifacts()

    def close_logger(self):
        if self._runner is not None:
            logger = self._runner.logger
            for handler in reversed(logger.handlers):
                handler.close()

    def set_train_val_summary_class(self, class_index, class_name):
        self._train_summary.set_class(class_index, class_name)
        self._train_result_summary.set_class(class_index, class_name)
        self._valid_result_summary.set_class(class_index, class_name)

    def set_summary_training_loss(self, epoch, loss):
        loss_meter = {'loss': loss}
        self._train_summary.add_tr_epoch_result(epoch, loss_meter)
        self._train_result_summary.summarize_result(epoch, loss=loss)

    def set_summary_validation_result(self, epoch, map, ap, mpe, mpe_by_class=None):
        self._train_summary.add_va_epoch_result(epoch, {})
        self._valid_result_summary.summarize_result(epoch, ap=ap, map=map, mpe=mpe, mpe_by_class=mpe_by_class)

    def _save_empty_records(self):
        self._train_summary.write_pkl(
            util.join_path(self._project_path, const.FILENAME_TRAIN_SUMMARY_PKL))
        self._train_result_summary.write_pkl(
            util.join_path(self._project_path, const.FILENAME_TRAIN_RESULT_SUMMARY_PKL))
        self._valid_result_summary.write_pkl(
            util.join_path(self._project_path, const.FILENAME_VALID_RESULT_SUMMARY_PKL))
        self._best_ckpt_measure.write_pkl(
            util.join_path(self._project_path, const.FILENAME_BEST_CKPT_MEASURE_PKL))

    def save_records_after_epoch(self, current_epoch, current_ckpt_path: str):
        checkpoint_store = ProjectCheckpointStore(self._project_path)
        save_results = [("last", self._save_ckpt(checkpoint_store, current_ckpt_path, const.FILENAME_CKPT_LAST))]
        is_best = self._check_if_the_ckpt_is_best(current_epoch)

        if is_best:
            save_results.append(("best", self._save_ckpt(checkpoint_store, current_ckpt_path, const.FILENAME_CKPT)))
            self._train_summary.add_model_update_epoch(current_epoch)

        self._train_summary.write_pkl(
            util.join_path(self._project_path, const.FILENAME_TRAIN_SUMMARY_PKL))
        self._train_result_summary.write_pkl(
            util.join_path(self._project_path, const.FILENAME_TRAIN_RESULT_SUMMARY_PKL))
        self._valid_result_summary.write_pkl(
            util.join_path(self._project_path, const.FILENAME_VALID_RESULT_SUMMARY_PKL))
        self._best_ckpt_measure.write_pkl(
            util.join_path(self._project_path, const.FILENAME_BEST_CKPT_MEASURE_PKL))

        save_summary = checkpoint_store.summarize(save_results)
        if save_summary.failed:
            TRAIN_LOGGER.error(
                "Epoch %d checkpoint save failed. succeeded=%s failed=%s",
                current_epoch,
                ", ".join(save_summary.succeeded) if save_summary.succeeded else "-",
                ", ".join(save_summary.failed),
            )
        else:
            TRAIN_LOGGER.info(
                "Epoch %d checkpoint saved (%s).",
                current_epoch,
                ", ".join(save_summary.succeeded),
            )

    def _load_records(self):
        self._train_summary.read_pkl(
            util.join_path(self._project_path, const.FILENAME_TRAIN_SUMMARY_PKL))
        self._train_result_summary.read_pkl(
            util.join_path(self._project_path, const.FILENAME_TRAIN_RESULT_SUMMARY_PKL))
        self._valid_result_summary.read_pkl(
            util.join_path(self._project_path, const.FILENAME_VALID_RESULT_SUMMARY_PKL))
        self._best_ckpt_measure.read_pkl(
            util.join_path(self._project_path, const.FILENAME_BEST_CKPT_MEASURE_PKL))

    def _save_ckpt(self, checkpoint_store: ProjectCheckpointStore, ckpt_path, save_dir):
        try:
            return checkpoint_store.save(ckpt_path, save_dir)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            error_msg = "ERROR - " + str(e)
            TRAIN_LOGGER.error(error_msg)
            return False

    def _clear_auto_saved(self):
        if os.path.exists(util.join_path(self._project_path, const.DIRNAME_AUTOSAVED)):
            try:
                shutil.rmtree(util.join_path(self._project_path, const.DIRNAME_AUTOSAVED))
            except Exception as e:
                traceback.print_tb(e.__traceback__)
                error_msg = "ERROR - " + str(e)
                APP_LOGGER.error(error_msg)

    def _clear_data_summary(self):
        if os.path.exists(self.get_dataset_summary_path()):
            util.remove_file(self.get_dataset_summary_path())

    def _check_if_the_ckpt_is_best(self, current_epoch: int) -> bool:
        best_map = self._best_ckpt_measure.get_map()
        current_map = self._valid_result_summary.get_metric(current_epoch, self.metric_name)

        best_training_loss = self._best_ckpt_measure.get_training_loss()
        current_training_loss = self._train_summary.tr_by_epoch[current_epoch]["loss"]

        best_mpe = self._best_ckpt_measure.get_mpe()
        current_mpe = self._valid_result_summary.get_metric(current_epoch, 'mPE')['corner_error']

        if current_map > best_map:
            self._best_ckpt_measure.set_map(current_map)
            self._best_ckpt_measure.set_training_loss(current_training_loss)
            self._best_ckpt_measure.set_mpe(current_mpe)
            self._best_ckpt_measure.set_epoch(current_epoch)
            return True
        elif current_map == best_map:
            if current_mpe < best_mpe:
                self._best_ckpt_measure.set_map(current_map)
                self._best_ckpt_measure.set_training_loss(current_training_loss)
                self._best_ckpt_measure.set_mpe(current_mpe)
                self._best_ckpt_measure.set_epoch(current_epoch)
                return True
            elif current_mpe == best_mpe:
                if current_training_loss < best_training_loss:
                    self._best_ckpt_measure.set_map(current_map)
                    self._best_ckpt_measure.set_training_loss(current_training_loss)
                    self._best_ckpt_measure.set_mpe(current_mpe)
                    self._best_ckpt_measure.set_epoch(current_epoch)
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def is_there_trained_checkpoint(self):
        if self._best_ckpt_measure.get_epoch() > 0:
            return True
        else:
            return False

    def get_dataset_summary_path(self):
        return os.path.join(self._project_path, "labeler_dataset_summary.json")

