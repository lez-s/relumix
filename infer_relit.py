import os  
import sys
sys.path.insert(0, os.getcwd())
import argparse
import copy
import cv2
import glob
import lovely_tensors
import multiprocessing as mp
import numpy as np
import os
import sys
import time
import tqdm
import tqdm.rich
import warnings
from einops import rearrange
from lovely_numpy import lo
from rich import print
from tqdm import TqdmExperimentalWarning
from huggingface_hub import hf_hub_download

# Internal imports.
from scripts import eval_utils

lovely_tensors.monkey_patch()
np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)


def test_args():

    parser = argparse.ArgumentParser()

    # Resource options.
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--debug', type=int, default=0)

    # General data options.
    parser.add_argument('--input', type=str, nargs='+',
                        help='One or more paths to video files, and/or directories with images, '
                        'and/or root of evaluation sets, and/or text files with list of examples.')
    parser.add_argument('--output', type=str,
                        default=r'../eval/output/relit')

    # General model options.
    parser.add_argument('--config_path', type=str)
    # Remove --model_path argument since we'll download from HuggingFace
    parser.add_argument('--hf_repo_id', type=str, default='lez/relumix',
                        help='Hugging Face repository ID for the model')
    parser.add_argument('--hf_filename', type=str, default='relumix.pth',
                        help='Model filename in the Hugging Face repository')
    parser.add_argument('--use_ema', type=int, default=0)
    parser.add_argument('--autocast', type=int, default=1)

    # Model inference options.
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=14)
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--guider_max_scale', type=float, default=1.5)
    parser.add_argument('--guider_min_scale', type=float, default=1.0)
    parser.add_argument('--motion_id', type=int, default=127)
    parser.add_argument('--force_custom_mbid', type=int, default=0)
    parser.add_argument('--cond_aug', type=float, default=0.02)
    parser.add_argument('--decoding_t', type=int, default=14)

    # Frame bounds options.
    parser.add_argument('--frame_start', type=int, default=0)
    parser.add_argument('--frame_stride', type=int, default=2)
    parser.add_argument('--frame_rate', type=int, default=12)

    # Relit specific options
    parser.add_argument('--ref_frame', type=str, default=None,
                        help='Path to pre-relighted first frame image or text file with list of reference frames')

    # Data processing options.
    parser.add_argument('--frame_width', type=int, default=512)
    parser.add_argument('--frame_height', type=int, default=512)
    parser.add_argument('--center_crop', type=int, default=1)
    parser.add_argument('--save_images', type=int, default=1)
    parser.add_argument('--save_mp4', type=int, default=1)
    parser.add_argument('--save_input', type=int, default=1)
    parser.add_argument('--vis_fps', type=int, default=15,
                       help='Frame rate for saved videos/visualizations')

    parser.add_argument('--is_file_list', type=int, default=0,
                       help='Whether input, ref_frame and output are text files containing lists')

    args = parser.parse_args()

    args.gpus = [int(x.strip()) for x in args.gpus.split(',')]

    return args


def load_input(args, worker_idx, example, model_bundle, ref_frame_path=None):
    '''
    NOTE: This method supports both known datasets as well as random input videos or images.
    :return input_rgb: (Tcm, 3, Hp, Wp) array of float32 in [-1, 1].
    :return controls (dict).
    :return batch (dict).
    '''
    import torch
    [model, train_config, test_config, device, model_name] = model_bundle[0:5]

    assert args.frame_start >= 0 and args.frame_stride >= 0 and args.frame_rate >= 0, \
        f'{args.frame_start} {args.frame_stride} {args.frame_rate}'

    # Simplify control parameters, only keep frame-related parameters
    controls = np.array([args.frame_start, args.frame_stride, args.frame_rate],
                        dtype=np.float32)

    # First check the actual frame count of the video
    # We use a temporary request to get the total frame count of the video
    video_frame_count = eval_utils.get_video_frame_count(example)
    print(f'[gray]{worker_idx}: Video has {video_frame_count} frames')
    
    # Pick all frames with contemporaneous input and output.
    Tc = args.num_frames
    orig_clip_frames = np.arange(Tc) * int(controls[1]) + int(controls[0])
    
    # Ensure clip_frames is always assigned
    clip_frames = orig_clip_frames.copy()
    
    # Ensure we don't request frames beyond the video range
    if video_frame_count > 0:  # If frame count was successfully obtained
        # Filter out frame indices that exceed the video length
        valid_indices = orig_clip_frames < video_frame_count
        if not np.all(valid_indices):
            print(f'[yellow]{worker_idx}: Warning: Some requested frames exceed video length. Truncating.')
            clip_frames = orig_clip_frames[valid_indices]
            # If all frames exceed the range, keep at least the first frame
            if len(clip_frames) == 0:
                clip_frames = np.array([min(int(controls[0]), max(0, video_frame_count-1))])
            # Update Tc to the actual number of frames used
            Tc = len(clip_frames)
    
    print(f'[gray]{worker_idx}: Tc: {Tc} clip_frames: {clip_frames}')
    assert np.all(clip_frames >= 0)

    # NOTE: If this is actually an image, it will be repeated as a still across all frames.
    try:
        input_rgb = eval_utils.load_image_or_video(
            example, clip_frames, args.center_crop, args.frame_width, args.frame_height, True)
    except ValueError as e:
        if "Video has" in str(e) and "frames, but requested" in str(e):
            # If frame count error still occurs, use fallback strategy
            print(f'[yellow]{worker_idx}: Error loading frames: {e}')
            print(f'[yellow]{worker_idx}: Falling back to safe loading strategy')
            
            # Recalculate safe clip_frames
            actual_frames = int(str(e).split('Video has ')[1].split(' frames')[0])
            safe_clip_frames = np.arange(min(Tc, actual_frames)) * int(controls[1]) + int(controls[0])
            safe_clip_frames = safe_clip_frames[safe_clip_frames < actual_frames]
            
            if len(safe_clip_frames) == 0:
                safe_clip_frames = np.array([0])
                
            Tc = len(safe_clip_frames)
            print(f'[yellow]{worker_idx}: Using safe frames: {safe_clip_frames}')
            
            input_rgb = eval_utils.load_image_or_video(
                example, safe_clip_frames, args.center_crop, args.frame_width, args.frame_height, True)
        else:
            # If it's another type of error, continue to raise
            raise
    
    # Handle frame count issue, ensure frame count matches args.num_frames
    actual_frames = input_rgb.shape[0]
    if (actual_frames < args.num_frames):
        # If insufficient frames, duplicate the last frame until reaching required frame count
        print(f'[yellow]{worker_idx}: Video has {actual_frames} frames, less than required {args.num_frames}. Padding with last frame.')
        last_frame = input_rgb[-1:].copy()  # Get and copy the last frame
        padding_frames = np.repeat(last_frame, args.num_frames - actual_frames, axis=0)
        input_rgb = np.concatenate([input_rgb, padding_frames], axis=0)
        # Update Tc to the corrected frame count
        Tc = args.num_frames
    
    input_rgb = (input_rgb + 1.0) / 2.0
    ori_input_rgb = copy.deepcopy(input_rgb)
    # (Tc, 3, Hp, Wp) array of float32 in [0, 1].
    
    # Use the passed ref_frame_path instead of args.ref_frame (if provided)
    ref_frame_path_to_use = ref_frame_path if ref_frame_path is not None else args.ref_frame
    
    # Initialize ref_frame as None
    ref_frame = None
        
    if ref_frame_path_to_use and os.path.exists(ref_frame_path_to_use):
        print(f'[cyan]{worker_idx}: Loading pre-relighted first frame from {ref_frame_path_to_use}...')
        ref_frame = eval_utils.load_relit_first_frame(
            ref_frame_path_to_use, args.center_crop, args.frame_width, args.frame_height, True)
        if (train_config.data.params.replace_first_frame == True):
            ref_frame = (ref_frame + 1.0) / 2.0
            input_rgb[0] = ref_frame
            print(f'[red]{worker_idx}: Replacing first frame with pre-relighted frame')
    else:
        raise ValueError(f'Invalid relit first frame path: {ref_frame_path_to_use}')

    batch = eval_utils.construct_batch_simple(
        input_rgb, Tc, args.frame_rate, args.motion_id, args.cond_aug,
        args.force_custom_mbid, model_bundle, device)
    batch['ori_input_rgb'] = (torch.from_numpy(ori_input_rgb)[None,...]* 2.0 - 1.0).to(device)
    if ref_frame is not None:
        batch['cond_frames_1dst'] = torch.from_numpy(ref_frame)[None,...].to(device)
        batch['cond_frames_1dst_with_noise'] = batch['cond_frames_1dst'] + args.cond_aug * torch.randn_like(batch['cond_frames_1dst'])

    (_, _, Hp, Wp) = input_rgb.shape
    assert Hp % 64 == 0 and Wp % 64 == 0, \
        f'Input resolution must be a multiple of 64, but got {Hp} x {Wp}'
    return (input_rgb, controls, batch, ref_frame)


def run_inference(args, device, model, batch):
    import torch

    autocast_kwargs = eval_utils.prepare_model_inference_params(
        model, device, args.num_steps, args.num_frames,
        args.guider_max_scale, args.guider_min_scale, args.autocast, args.decoding_t)

    with torch.no_grad():
        with torch.autocast(**autocast_kwargs):
            pred_samples = []

            for sample_idx in range(args.num_samples):    
                # 执行完整视频的去噪循环
                video_dict = model.sample_video(
                    batch, enter_ema=False, limit_batch=False)

                output_dict = dict()
                output_dict['cond_rgb'] = video_dict['cond_video'].detach().cpu().numpy()
                # (Tcm, 3, Hp, Wp) = (14, 3, 256, 384) array of float32 in [0, 1].
                output_dict['sampled_rgb'] = video_dict['sampled_video'].detach().cpu().numpy()
                # (Tcm, 3, Hp, Wp) = (14, 3, 256, 384) array of float32 in [0, 1].
                output_dict['sampled_latent'] = video_dict['sampled_z'].detach().cpu().numpy()
                # (Tcm, 4, Hl, Wl) = (14, 4, 32, 48) array of float32.
                pred_samples.append(output_dict)

    return pred_samples


def get_controls_friendly(controls):
    frame_start = int(controls[0])
    frame_stride = int(controls[1])
    frame_rate = int(controls[2])

    title = f'Frame Rate {frame_rate}'
    filename = f'fs{frame_start}_fr{frame_rate}'

    return (True, title, filename)


def create_visualizations(
        args, input_rgb, controls_friendly, pred_samples,
        model_name, ref_frame=None, ori_input_rgb=None):
    '''
    :param input_rgb: (Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    :param pred_samples: List of dict.
    :param ori_input_rgb: (Tcm, 3, Hp, Wp) array of float32 in [0, 1], original input before modification.
    '''
    (Tcm, _, Hp, Wp) = input_rgb.shape
    S = len(pred_samples)

    # Use original input images for visualization, if provided
    if ori_input_rgb is not None:
        input_rgb_vis = ori_input_rgb
    else:
        input_rgb_vis = input_rgb
        
    input_rgb_vis = rearrange(input_rgb_vis, 't c h w -> t h w c')
    # (Tcm, Hp, Wp, 3) array of float32 in [0, 1].

    pred_samples_rgb = []

    if S >= 1:
        pred_samples_rgb = [
            rearrange(x['sampled_rgb'], 't c h w -> t h w c') for x in pred_samples]
        # (S, Tcm, Hp, Wp, 3) array of float32 in [0, 1].

    # Prepare first frame comparison for visualization
    first_frame_comparison = False
    if ref_frame is not None:
        first_frame_comparison = True
        ref_frame = rearrange(ref_frame, 'c h w -> h w c')

    # Create visualization layouts
    rich1_frames = []
    rich2_frames = []
    rich3_frames = []  # Add layout for showing first frame relighting
    font_size = 1.0

    
    for t in range(Tcm):
        # Layout 1: Input || Output Relit
        canvas1 = np.zeros((Hp + 40, Wp * 2, 3), dtype=np.float32)

        eval_utils.draw_text(canvas1, (20, 5), (0.5, 0.0),
                            f'Input (Frame {t})', (1, 1, 1), font_size)
        eval_utils.draw_text(canvas1, (Wp + 20, 5), (0.5, 0.0),
                            f'Relit Output', (1, 1, 1), font_size)

        canvas1[40:Hp + 40, 0:Wp] = input_rgb_vis[t]
        
        if S >= 1:
            canvas1[40:Hp + 40, Wp:Wp * 2] = pred_samples_rgb[0][t].copy()

        rich1_frames.append(canvas1)
        
        # Rich 2: Side-by-side comparison with detail
        if S >= 1:
            canvas2 = np.zeros((Hp * 2 + 80, Wp * 2, 3), dtype=np.float32)
            
            eval_utils.draw_text(canvas2, (20, 5), (0.5, 0.0),
                                f'Input (Frame {t})', (1, 1, 1), font_size)
            eval_utils.draw_text(canvas2, (Wp + 20, 5), (0.5, 0.0),
                                f'Relit Output', (1, 1, 1), font_size)
            
            canvas2[40:Hp + 40, 0:Wp] = input_rgb_vis[t]
            canvas2[40:Hp + 40, Wp:Wp * 2] = pred_samples_rgb[0][t].copy()
            
            # Add an enlarged view of specific regions if needed
            if t % 4 == 0:  # Only show for some frames
                zoom_region_h = slice(Hp//3, Hp//3*2)
                zoom_region_w = slice(Wp//3, Wp//3*2)
                
                zoom_factor = 2
                zoom_h = (zoom_region_h.stop - zoom_region_h.start) * zoom_factor
                zoom_w = (zoom_region_w.stop - zoom_region_w.start) * zoom_factor
                
                eval_utils.draw_text(canvas2, (Hp + 60, 5), (0.5, 0.0),
                                    f'Input Detail', (1, 1, 1), font_size)
                eval_utils.draw_text(canvas2, (Hp + 60, Wp + 5), (0.5, 0.0),
                                    f'Relit Detail', (1, 1, 1), font_size)
                
                # Resize region for detail view
                input_detail = cv2.resize(input_rgb_vis[t][zoom_region_h, zoom_region_w], 
                                        (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
                output_detail = cv2.resize(pred_samples_rgb[0][t][zoom_region_h, zoom_region_w], 
                                        (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
                
                canvas2[Hp + 80:Hp + 80 + zoom_h, 0:zoom_w] = input_detail
                canvas2[Hp + 80:Hp + 80 + zoom_h, Wp:Wp + zoom_w] = output_detail
            
            rich2_frames.append(canvas2)

    # Organize & return results.
    vis_dict = dict()

    # Pause a tiny bit at the beginning and end for less jerky looping.
    rich1_frames = [rich1_frames[0]] + rich1_frames + [rich1_frames[-1]] * 2
    rich1_frames = np.stack(rich1_frames, axis=0)
    rich1_frames = np.clip(rich1_frames, 0.0, 1.0)
    vis_dict['rich1'] = rich1_frames

    if len(rich2_frames) > 0:
        rich2_frames = [rich2_frames[0]] + rich2_frames + [rich2_frames[-1]] * 2
        rich2_frames = np.stack(rich2_frames, axis=0)
        rich2_frames = np.clip(rich2_frames, 0.0, 1.0)
        vis_dict['rich2'] = rich2_frames
        
    if len(rich3_frames) > 0:
        # Convert to numpy array
        rich3_frames = np.stack(rich3_frames, axis=0)
        rich3_frames = np.clip(rich3_frames, 0.0, 1.0)
        vis_dict['rich3'] = rich3_frames

    vis_dict['input'] = input_rgb_vis
    vis_dict['output'] = pred_samples_rgb
    

    return vis_dict


def save_results(args, vis_dict, controls, output_fp1, output_fp2):
    vis_fps = args.vis_fps

    eval_utils.write_video_and_frames(
        vis_dict['rich1'], dst_dp=output_fp1 + '_gal', fps=vis_fps,
        save_images=False, save_mp4=True, quality=9)

    # if 'rich2' in vis_dict:
    #     eval_utils.write_video_and_frames(
    #         vis_dict['rich2'], dst_dp=output_fp1 + '_detail', fps=vis_fps,
    #         save_images=False, save_mp4=True, quality=9)
            
    # Save first frame comparison image
    if 'rich3' in vis_dict:
        eval_utils.write_video_and_frames(
            vis_dict['rich3'], dst_dp=output_fp1 + '_firstframe', fps=1,
            save_images=True, save_mp4=False, quality=9)

    if args.save_images or args.save_mp4:
        if args.save_input:
            eval_utils.write_video_and_frames(
                vis_dict['input'], dst_dp=output_fp2 + '_input', fps=vis_fps,
                save_images=args.save_images, save_mp4=args.save_mp4, quality=9)

        for s in range(args.num_samples):
            eval_utils.write_video_and_frames(
                vis_dict['output'][s], dst_dp=output_fp2 + f'_relit_s{s}', fps=vis_fps,
                save_images=args.save_images, save_mp4=args.save_mp4, quality=9)


def process_example(args, worker_idx, example_idx, example, model_bundle, ref_frame_path=None, output_path=None):
    (model, train_config, test_config, device, model_name) = model_bundle[0:5]

    # Prepare output directory first to check if results already exist
    # If a specific output path is provided, use it
    if output_path:
        output_fp1 = output_path
        output_fp2 = os.path.join(output_path, 'extra')
        check_file = os.path.join(output_fp1, f'video.mp4')
    else:
        test_tag = os.path.basename(args.output).split('_')[0]
        output_fn = os.path.splitext(os.path.basename(example))[0]
        output_fn = output_fn.replace('_p0', '')
        output_fn = output_fn.replace('_rgb', '')

        output_fn1 = f'{test_tag}_{example_idx:03d}_n{model_name}'
        output_fn1 += f'_{output_fn}'
        output_fn2 = output_fn1  # Save shorter name for extra data.

        # Prepare controls friendly name for filename
        controls = np.array([args.frame_start, args.frame_stride, args.frame_rate],
                             dtype=np.float32)
        controls_friendly = get_controls_friendly(controls)
        
        # Contains either just frame rate or frame rate + camera controls + light info.
        output_fn1 += f'_{controls_friendly[2]}'

        # For more prominent / visible stuff:
        output_fp1 = os.path.join(args.output, output_fn1)

        # For less prominent but still useful stuff:
        output_fp2 = os.path.join(args.output, 'extra', output_fn2)
        check_file = f'{output_fp1}_gal.mp4'
    
    # Check if output file already exists, skip processing if it does
    if os.path.exists(check_file) and not args.debug:
        print(f'[yellow]{worker_idx}: Output file {check_file} already exists, skipping...')
        return True
    
    # Create output directories
    if output_path:
        os.makedirs(output_fp1, exist_ok=True)
        os.makedirs(output_fp2, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(output_fp1), exist_ok=True)
        os.makedirs(os.path.dirname(output_fp2), exist_ok=True)

    # Load & preprocess input frames.
    print()
    print(f'[yellow]{worker_idx}: Loading input frames from {example}...')
    start_time = time.time()
    (input_rgb, controls, batch, ref_frame) = load_input(
        args, worker_idx, example, model_bundle, ref_frame_path)
    print(f'[magenta]{worker_idx}: Loading frames took {time.time() - start_time:.2f}s')

    # Get original input images for visualization
    ori_input_rgb = batch['ori_input_rgb'].cpu().numpy()[0]
    ori_input_rgb = (ori_input_rgb + 1.0) / 2.0  # Normalize from [-1,1] to [0,1]

    # Run inference.
    print()
    print(f'[cyan]{worker_idx}: Running Relit model on selected video clip...')
    start_time = time.time()
    if args.num_samples >= 1:
        pred_samples = run_inference(
            args, device, model, batch)
    else:
        pred_samples = []
    print(f'[magenta]{worker_idx}: Inference took {time.time() - start_time:.2f}s')

    # Create rich inference visualization.
    print()
    print(f'[yellow]{worker_idx}: Creating rich visualizations...')
    start_time = time.time()
    controls_friendly = get_controls_friendly(controls)
    vis_dict = create_visualizations(
        args, input_rgb, controls_friendly, pred_samples,
        model_name, ref_frame, ori_input_rgb)
    print(f'[magenta]{worker_idx}: Visualizations took {time.time() - start_time:.2f}s')

    # Save results to disk.
    print()
    print(f'[yellow]{worker_idx}: Saving results to disk...')
    start_time = time.time()
    save_results(args, vis_dict, controls, output_fp1, output_fp2)
    print(f'[magenta]{worker_idx}: Saving took {time.time() - start_time:.2f}s')

    return True


def worker_fn(args, worker_idx, num_workers, gpu_idx, example_list):

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)

    # Only now can we import torch.
    import torch
    from sgm.util import instantiate_from_config
    torch.set_printoptions(precision=3, sci_mode=False, threshold=1000)

    # Update CPU affinity.
    eval_utils.update_os_cpu_affinity(worker_idx, num_workers)

    # Download model from Hugging Face
    print(f'[cyan]{worker_idx}: Downloading model from Hugging Face repository {args.hf_repo_id}...')
    start_time = time.time()
    
    try:
        model_path = hf_hub_download(
            repo_id=args.hf_repo_id,
            filename=args.hf_filename,
            cache_dir=os.path.expanduser("~/.cache/huggingface")
        )
        print(f'[green]{worker_idx}: Model downloaded to {model_path}')
        # model_path = 'logs/relumix.pth'
    except Exception as e:
        print(f'[red]{worker_idx}: Failed to download model from Hugging Face: {e}')
        raise

    print(f'[magenta]{worker_idx}: Model download took {time.time() - start_time:.2f}s')

    print()
    print(f'[cyan]{worker_idx}: Loading Relit model from {model_path} on GPU {gpu_idx}...')
    start_time = time.time()

    device = args.device
    if 'cuda' in device:
        device = f'cuda:{gpu_idx}'

    # Initialize model.
    model_bundle = eval_utils.load_model_bundle(
        device, args.config_path, model_path, args.use_ema,
        num_steps=args.num_steps, num_frames=args.num_frames,
        max_scale=args.guider_max_scale, min_scale=args.guider_min_scale,
        verbose=(worker_idx == 0))
    (model, train_config, test_config, device, model_name) = model_bundle[0:5]

    print(f'[magenta]{worker_idx}: Loading Relit model took {time.time() - start_time:.2f}s')

    eval_utils.warn_resolution_mismatch(train_config, args.frame_width, args.frame_height)

    # If in file list mode, load reference frame and output path lists
    ref_frame_list = None
    output_path_list = None
    
    if args.is_file_list:
        # Load reference frame list
        if args.ref_frame and os.path.exists(args.ref_frame):
            with open(args.ref_frame, 'r') as f:
                ref_frame_list = [line.strip() for line in f.readlines()]
                print(f'[cyan]{worker_idx}: Loaded {len(ref_frame_list)} reference frames from {args.ref_frame}')
        
        # Load output path list
        if args.output and os.path.exists(args.output):
            with open(args.output, 'r') as f:
                output_path_list = [line.strip() for line in f.readlines()]
                print(f'[cyan]{worker_idx}: Loaded {len(output_path_list)} output paths from {args.output}')
    
    # Start iterating over all videos.
    if args.debug:
        to_loop = tqdm.tqdm(list(enumerate(example_list)))
    else:
        to_loop = tqdm.rich.tqdm(list(enumerate(example_list)))

    # Enable EMA scope early on to avoid constant shifting around of weights.
    with model.ema_scope('Testing'):

        for (i, example) in to_loop:
            example_idx = i * num_workers + worker_idx
            
            # If in file list mode, use corresponding index's reference frame and output path
            ref_frame_path = None
            output_path = None
            
            if args.is_file_list:
                if ref_frame_list and example_idx < len(ref_frame_list):
                    ref_frame_path = ref_frame_list[example_idx]
                    
                if output_path_list and example_idx < len(output_path_list):
                    output_path = output_path_list[example_idx]
                    os.makedirs(output_path, exist_ok=True)

            process_example(args, worker_idx, example_idx, example, model_bundle, 
                           ref_frame_path, output_path)

    print()
    print(f'[cyan]{worker_idx}: Done!')
    print()


def main(args):

    # Create output directory
    if not args.is_file_list:
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(os.path.join(args.output, 'extra'), exist_ok=True)
    
    # Validate the first frame for relighting
    if args.ref_frame and not args.is_file_list and not os.path.exists(args.ref_frame):
        print(f'[red]Warning: Specified relit first frame {args.ref_frame} does not exist!')

    # Save the arguments to this inference script.
    if not args.is_file_list:
        args_fp = os.path.join(args.output, 'args_infer_relit.json')
        eval_utils.save_json(vars(args), args_fp)
        print(f'[yellow]Saved script args to {args_fp}')

    # Load list of videos to process (not the pixels themselves yet).
    print()
    print(f'[yellow]Parsing list of individual examples from {args.input}...')
    start_time = time.time()

    examples = eval_utils.get_list_of_input_images_or_videos(args.input)
    print(f'[yellow]Found {len(examples)} examples '
          f'(counting both video files and/or image folders).')

    print(f'[magenta]Loading data list took {time.time() - start_time:.2f}s')

    assert len(examples) > 0, f'No examples found in {args.input}!'

    # Split examples across workers (simplified since we only have one model now).
    num_gpus = len(args.gpus)
    num_buckets = num_gpus
    worker_args_list = []

    for worker_idx in range(num_gpus):
        # Every GPU processes a different subset of examples
        gpu_idx = args.gpus[worker_idx % num_gpus]
        bucket_idx = worker_idx

        my_examples = examples[bucket_idx::num_buckets]
        worker_args_list.append((args, worker_idx, num_gpus, gpu_idx, my_examples))

    print(f'[cyan]Splitting {len(examples)} examples across {num_gpus} workers '
          f'according to specified GPU devices: {args.gpus}...')
    start_time = time.time()

    if num_gpus > 1:
        print(f'[cyan]Starting {num_gpus} processes...')

        import torch
        import torch.multiprocessing
        torch.multiprocessing.set_start_method('spawn')

        with mp.Pool(processes=num_gpus) as pool:
            results = pool.starmap(worker_fn, worker_args_list)

    else:
        print(f'[cyan]Calling method directly...')
        results = []
        for worker_args in worker_args_list:
            results.append(worker_fn(*worker_args))

    print(f'[magenta]Everything took {time.time() - start_time:.2f}s')

    print()
    print(f'[cyan]Done!')
    print()


if __name__ == '__main__':

    args = test_args()

    main(args)

    pass
