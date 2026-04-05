from mmdeploy.backend.openvino.backend_manager import OpenVINOManager
from subprocess import PIPE, CalledProcessError, run
from typing import Dict, Optional, Sequence, Union, Any
import logging

class DiceOpenVINOManager(OpenVINOManager):
    @classmethod
    def to_backend(cls,
                   ir_files: Sequence[str],
                   work_dir: str,
                   deploy_cfg: Any,
                   log_level: int = logging.INFO,
                   device: str = 'cpu',
                   **kwargs) -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            ir_files (Sequence[str]): The intermediate representation files.
            work_dir (str): The work directory, backend files and logs should
                be saved in this directory.
            deploy_cfg (Any): The deploy config.
            log_level (int, optional): The log level. Defaults to logging.INFO.
            device (str, optional): The device type. Defaults to 'cpu'.
        Returns:
            Sequence[str]: Backend files.
        """
        from mmdeploy.backend.openvino import is_available
        from aligner_engine.mm_deploy.dice.dice_onnx2openvino import from_onnx

        assert is_available(), \
            'OpenVINO is not available, please install OpenVINO first.'

        from mmdeploy.apis.openvino import (get_input_info_from_cfg,
                                            get_mo_options_from_cfg,
                                            get_output_model_file)
        from mmdeploy.utils import get_ir_config

        openvino_files = []
        for onnx_path in ir_files:
            model_xml_path = get_output_model_file(onnx_path, work_dir)
            input_info = get_input_info_from_cfg(deploy_cfg)
            output_names = get_ir_config(deploy_cfg).output_names
            mo_options = get_mo_options_from_cfg(deploy_cfg)
            from_onnx(onnx_path, work_dir, input_info, output_names,
                      mo_options)
            openvino_files.append(model_xml_path)

        return openvino_files

