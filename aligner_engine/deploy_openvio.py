
import logging
import os.path as osp
import os
from copy import deepcopy
import aligner_engine.const as const
import aligner_engine.utils as util


def deploy_openvino(model_cfg_path, checkpoint_path, sample_img_path, work_dir, processing_signal):
    import mmengine
    from mmdeploy.apis import torch2onnx
    from mmdeploy.utils import Backend, get_backend, get_ir_config, load_config
    from aligner_engine.mm_deploy.dice.dice_openvino_manager import DiceOpenVINOManager

    device = "cpu"
    uri = "192.168.1.1:60000"

    # create work_dir if not
    mmengine.mkdir_or_exist(osp.abspath(work_dir))

    deploy_cfg_base_path = util.join_path(util.ROOT_PATH, "aligner_engine", "mm_rotate_det", "dice",
                                             "configs", "deploy", "mmrotate",
                                             "rotated-detection-dice-openvino-dynamic-512x512.py")


    # load deploy_cfg
    deploy_cfg_base, model_cfg = load_config(deploy_cfg_base_path, model_cfg_path)

    # modify deploy_cfg
    deploy_cfg_path = str(os.path.join(work_dir, const.FILENAME_DEPLOY_CONFIG))
    deploy_cfg = deepcopy(deploy_cfg_base)
    test_pipeline = model_cfg.test_dataloader.dataset.pipeline

    rescale_value = (0, 0)
    for idx, transform in enumerate(test_pipeline):
        if transform.type == "mmdet.Resize":
            rescale_value = transform.scale
            break

    deploy_cfg.backend_config.model_inputs[0].opt_shapes.input = [1, 3, rescale_value[0], rescale_value[1]]

    deploy_cfg.dump(deploy_cfg_path)
    # convert to IR
    ir_config = get_ir_config(deploy_cfg)
    ir_save_file = ir_config['save_file']
    # ir_type = IR.get(ir_config['type'])

    processing_signal.emit(2, 'Building intermediate representation...')
    torch2onnx(
        sample_img_path,
        work_dir,
        ir_save_file,
        deploy_cfg_path,
        model_cfg_path,
        checkpoint_path,
        device=device)

    # convert backend
    ir_files = [osp.join(work_dir, ir_save_file)]
    backend_files = ir_files
    # convert backend
    backend = get_backend(deploy_cfg)

    # convert to backend
    processing_signal.emit(3, 'Optimizing model...')
    backend_files = DiceOpenVINOManager.to_backend(
        ir_files,
        work_dir=work_dir,
        deploy_cfg=deploy_cfg,
        log_level="INFO",
        device=device,
        uri=uri)

    # draw each result
    processing_signal.emit(4, 'Validating...')
    extra = dict(
        backend=backend,
        output_file=osp.join(work_dir, f'output_{backend.value}.jpg'),
        show_result=False)
    try:
        from aligner_engine.mm_deploy.dice.visualize import visualize_model
    except ImportError:
        visualize_model = None

    if visualize_model is not None:
        visualize_model(model_cfg_path, deploy_cfg_path, backend_files, sample_img_path, device, **extra)

        extra_torch = dict(
                backend=Backend.PYTORCH,
                output_file=osp.join(work_dir, 'output_pytorch.jpg'),
                show_result=False)
        visualize_model(model_cfg_path, deploy_cfg_path, [checkpoint_path], sample_img_path, device, **extra_torch)
    else:
        logging.warning("visualize_model is unavailable. Skipping export validation preview.")
