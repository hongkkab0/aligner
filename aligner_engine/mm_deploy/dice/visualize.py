import copy
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
from mmdeploy.codebase.mmrotate.deploy.rotated_detection import RotatedDetection
from mmengine import Config, Registry
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
import torch
from torch import nn
from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, load_config)
from mmdeploy.codebase.mmrotate.deploy.rotated_detection_model import End2EndModel
from aligner_engine.mm_deploy.dice.dice_open_vino_wrapper import DiceOpenVINOWrapper


class DiceRotatedDetection(RotatedDetection):
    def build_backend_model(self,
                            model_files: Optional[str] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        print('DiceRotatedDetection')
        data_preprocessor = copy.deepcopy(
            self.model_cfg.model.get('data_preprocessor', {}))
        data_preprocessor.setdefault('type', 'mmdet.DetDataPreprocessor')

        model = build_rotated_detection_model(
            model_files,
            self.deploy_cfg,
            device=self.device,
            data_preprocessor=data_preprocessor)
        model = model.to(self.device)
        return model.eval()

    def get_visualizer(self, name: str, save_dir: str):
        visualizer = super().get_visualizer(name, save_dir)
        visualizer.dataset_meta = {}
        return visualizer


class DiceEnd2EndModel(End2EndModel):
    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str):
        """Initialize the wrapper of backends.

        Args:
            backend (Backend): The backend enum, specifying backend type.
            backend_files (Sequence[str]): Paths to all required backend files
                (e.g. .onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string represents device type.
        """
        output_names = self.output_names
        self.wrapper = DiceOpenVINOWrapper(ir_model_file=backend_files[0], output_names=output_names)


def build_rotated_detection_model(
        model_files: Sequence[str],
        deploy_cfg: Union[str, Config],
        device: str,
        data_preprocessor: Optional[Union[Config,
                                          BaseDataPreprocessor]] = None,
        **kwargs):
    """Build rotated detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | Config): Input model config file or Config
            object.
        deploy_cfg (str | Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Rotated detector for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, = load_config(deploy_cfg)
    backend = get_backend(deploy_cfg)
    backend_rotated_detector = DiceEnd2EndModel(
            backend=backend,
            backend_files=model_files,
            device=device,
            deploy_cfg=deploy_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs)

    return backend_rotated_detector