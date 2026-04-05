_base_ = ['./rotated-detection_dynamic.py', '../_base_/backends/openvino.py']

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 512, 512]))],
    mo_options=dict(flags=["--compress_to_fp16"])
)
