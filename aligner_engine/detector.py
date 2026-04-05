from __future__ import annotations

import os
import numpy as np
from typing import Optional, Union
import warnings
from pathlib import Path


class Detector:
    def __init__(self,
                 model_config_path: str,
                 checkpoint_path: str,
                 device: str,
                 deploy_config_path: str = "",
                 vino_xml_path: str = "",
                 enable_vino: bool = True):
        from mmengine.config import Config
        from mmrotate.utils import register_all_modules

        # register all modules in mmrotate into the registries
        register_all_modules()
        import aligner_engine.mm_rotate_det.dice  # to register dice modules

        # build the model from a config file and a checkpoint file
        self._is_model_on_vino = False
        if device == "cpu":
            can_use_vino = (
                enable_vino
                and bool(deploy_config_path)
                and bool(vino_xml_path)
                and os.path.exists(deploy_config_path)
                and os.path.exists(vino_xml_path)
            )
            if can_use_vino:
                from mmdeploy.utils.config_utils import load_config
                from aligner_engine.mm_deploy.dice.dice_rotated_detection import DiceRotatedDetection

                deploy_cfg, model_cfg = load_config(deploy_config_path, model_config_path)
                task_processor = DiceRotatedDetection(model_cfg, deploy_cfg, device)
                self._model = task_processor.build_backend_model(
                    [vino_xml_path],
                    data_preprocessor_updater=task_processor.update_data_preprocessor)
                self._is_model_on_vino = True
                print("Model is loaded on fast CPU.")
            else:
                self._model = self._init_detector(model_config_path, checkpoint_path, device=device)
                print("Model is loaded on slow CPU.")
        else:
            self._model = self._init_detector(model_config_path, checkpoint_path, device=device)
            print("Model is loaded on GPU.")

        self._pipeline = None
        self._model_config = Config.fromfile(model_config_path)
        self._classes = self._model_config.classes_dice
        self._init_pipeline()

    def _init_pipeline(self):
        from mmdet.utils import get_test_pipeline_cfg
        from mmcv.transforms import Compose

        cfg = self._model_config.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self._pipeline = Compose(test_pipeline)

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._init_pipeline()
        return self._pipeline

    def inference(self, img: np.ndarray):
        from aligner_engine.mm_rotate_det.dice.DICErbox2qbox import dice_rbox_to_qbox_single_with_sort_rule

        data_ = dict(img=img, img_id=0)
        data_ = self.pipeline(data_)
        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]
        result = self._model.test_step(data_)[0]

        predictions = {}
        for pred_idx, pred in enumerate(result.pred_instances):
            label = pred.labels.detach().cpu().item()
            class_name = self._classes[label]
            if self._is_model_on_vino:
                qbox, _ = dice_rbox_to_qbox_single_with_sort_rule(pred.bboxes.tensor)
                qbox = qbox.detach().cpu().tolist()
            else:
                qbox, _ = dice_rbox_to_qbox_single_with_sort_rule(pred.bboxes)
                qbox = qbox.detach().cpu().tolist()
            predictions[str(pred_idx)] = dict(
                class_name=class_name,
                conf=pred.scores.detach().cpu().item(),
                qbox=qbox)  # [x1 y1 x2 y2 x3 y3 x4 y4]

        return predictions

    def unload_model(self):
        import torch

        self._model.cpu()
        del self._model
        torch.cuda.empty_cache()

    def to_cpu(self):
        self._model.cpu()

    def _init_detector(self,
                       config: Union[str, Path, Config],
                       checkpoint: Optional[str] = None,
                       palette: str = 'none',
                       device: str = 'cuda:0',
                       cfg_options: Optional[dict] = None,
                       ):
        from mmengine.config import Config
        from mmengine.model.utils import revert_sync_batchnorm
        from mmengine.registry import init_default_scope
        from mmengine.runner import load_checkpoint
        from mmdet.evaluation import get_classes
        from mmdet.registry import MODELS

        """Initialize a detector from config file.

        Args:
            config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
                :obj:`Path`, or the config object.
            checkpoint (str, optional): Checkpoint path. If left as None, the model
                will not load any weights.
            palette (str): Color palette used for visualization. If palette
                is stored in checkpoint, use checkpoint's palette first, otherwise
                use externally passed palette. Currently, supports 'coco', 'voc',
                'citys' and 'random'. Defaults to none.
            device (str): The device where the anchors will be put on.
                Defaults to cuda:0.
            cfg_options (dict, optional): Options to override some settings in
                the used config.

        Returns:
            nn.Module: The constructed detector.
        """
        if isinstance(config, (str, Path)):
            config = Config.fromfile(config)
        elif not isinstance(config, Config):
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(config)}')
        if cfg_options is not None:
            config.merge_from_dict(cfg_options)
        elif 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None
        init_default_scope(config.get('default_scope', 'mmdet'))

        model = MODELS.build(config.model)
        model = revert_sync_batchnorm(model)
        if checkpoint is None:
            warnings.simplefilter('once')
            warnings.warn('checkpoint is None, use COCO classes by default.')
            model.dataset_meta = {'classes': get_classes('coco')}
        else:
            checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
            # Weights converted from elsewhere may not have meta fields.
            checkpoint_meta = checkpoint.get('meta', {})

            # save the dataset_meta in the model for convenience
            if 'dataset_meta' in checkpoint_meta:
                # mmdet 3.x, all keys should be lowercase
                model.dataset_meta = {
                    k.lower(): v
                    for k, v in checkpoint_meta['dataset_meta'].items()
                }
            elif 'CLASSES' in checkpoint_meta:
                # < mmdet 3.x
                classes = checkpoint_meta['CLASSES']
                model.dataset_meta = {'classes': classes}
            else:
                warnings.simplefilter('once')
                warnings.warn(
                    'dataset_meta or class names are not saved in the '
                    'checkpoint\'s meta data, use COCO classes by default.')
                model.dataset_meta = {'classes': get_classes('coco')}

        model.cfg = config  # save the config in the model for convenience
        model.to(device)
        model.eval()
        return model
