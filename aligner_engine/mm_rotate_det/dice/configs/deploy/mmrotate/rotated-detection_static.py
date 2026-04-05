_base_ = ['../_base_/onnx_config.py']

onnx_config = dict(output_names=['dets', 'labels'], input_shape=None)
codebase_config = dict(
    type='mmrotate',
    task='RotatedDetection',
    post_processing=dict(
        score_threshold=0.05,
        iou_threshold=0.1,
        pre_top_k=3000,
        keep_top_k=2000,
        max_output_boxes_per_class=2000))
