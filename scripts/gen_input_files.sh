#!/bin/bash

BASE_DIR=./carla_relumix
input_video_base_path=./carla_relit/split_videos
input_video_list=(Town02/scene_000/morning_clear
            Town10HD/scene_010/bright_sun
            Town10HD/scene_023/bright_sun
            Town06/scene_000/morning_clear
            Town06/scene_002/bright_sun
            Town10HD/scene_033/afternoon_cloudy
            Town06/scene_001/morning_light_clouds
)
ref_video_list=(Town02/scene_000
            Town10HD/scene_010
            Town10HD/scene_023
            Town06/scene_000
            Town06/scene_002
            Town10HD/scene_033
            Town06/scene_001
)

output_files_path=$BASE_DIR/relumix_input_files
output_root_dir=$BASE_DIR/relumix_output
mkdir -p "$output_files_path"
mkdir -p "$output_root_dir"

# Create empty files
> "$output_files_path/input_video_path.txt"
> "$output_files_path/ref_frame_path.txt"
> "$output_files_path/output_path.txt"

for i in "${!input_video_list[@]}"; do
    scene=${input_video_list[i]}
    ref_scene_base=${ref_video_list[i]}
    
    # Find all subdirectories with mp4 files in the reference scene
    for ref_subdir in $(find $input_video_base_path/${ref_scene_base} -type d -name "*segments"); do
        ref_subdir_name=$(basename "$ref_subdir")
        ref_condition_name=${ref_subdir_name%_segments}
        
        # read $input_video_base_path/${scene}_segments/{segment_0001-0025}.mp4 and loop through each segment
        segment_count=0
        for segment in $(ls $input_video_base_path/${scene}_segments/*.mp4 | sort | head -10); do
            file_name=$(basename "$segment")
            input_video_path="${input_video_base_path}/${scene}_segments/${file_name}"

            ref_video_path="${ref_subdir}/${file_name}"
            ref_frame_path="${input_video_base_path}/ref_frame/${ref_scene_base}/${ref_condition_name}/${file_name%.mp4}.png"

            
            mkdir -p "$(dirname "$ref_frame_path")"
            # extract 1st frame from ref video and save to ref_frame_path
            # ffmpeg -y -i "$ref_video_path" -vf "select=eq(n\,0)" -q:v 3 "$ref_frame_path"

            output_path="${output_root_dir}/${scene}2${ref_condition_name}/${file_name%.mp4}_relit"

            echo "$input_video_path" >> "$output_files_path/input_video_path.txt"
            echo "$ref_frame_path" >> "$output_files_path/ref_frame_path.txt"
            echo "$output_path" >> "$output_files_path/output_path.txt"
        done
    done
done