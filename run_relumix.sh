yaml_config=infer_relumix.yaml
input_txt=examples/input_video_path.txt
output_txt=examples/output_path.txt
ref_txt=examples/ref_frame_path.txt


python infer_relit.py --gpus=0 \
    --config_path=carla_config/$yaml_config \
    --input=$input_txt \
    --ref_frame=$ref_txt \
    --output=$output_txt \
    --is_file_list=1 \
    --frame_start=0 --frame_stride=1 --frame_rate=15 --num_frames=61 \
    --autocast=1 --num_samples=1 --num_steps=25 