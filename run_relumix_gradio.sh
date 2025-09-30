#!/bin/bash

yaml_config=infer_relumix_gradio.yaml
hf_repo_id='lez/relumix'
hf_filename='relumix.pth'
output_path='../eval_output/relight/'
task_desc='Video Relighting with Reference Frame'

CUDA_VISIBLE_DEVICES=0 python gradio_app.py --device=cuda \
    --config_path=carla_config/$yaml_config \
    --hf_repo_id=${hf_repo_id} \
    --hf_filename=${hf_filename} \
    --output_path=${output_path} \
    --task_desc="${task_desc}" \
    --input_frames=14 \
    --resolution='512 x 512' \
    --task_mode=relight \
    --port=7881 \
    --debug=False
