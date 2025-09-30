import os  
import sys 
sys.path.insert(0, os.getcwd()) 

# Library imports.
import fire
import gradio as gr
import glob
import numpy as np
import os
import time
import torch
from einops import rearrange
from functools import partial
from rich import print
from huggingface_hub import hf_hub_download

# Internal imports.
from scripts import eval_utils
from sgm.data import common

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False, threshold=1000)

_TITLE = '[{model_name}] Video Relighting with Reference Frame'

_DESCRIPTION = '''
This demo showcases Video Relighting capabilities.
We use a finetuned video diffusion model to relight videos based on a reference frame.

Upload a video and a reference frame showing the desired lighting condition,
and the model will generate a relit version of the entire video.

All results are saved to your disk for reproducibility and debugging purposes.

The currently loaded checkpoint path is `{model_path}`, and performs the following task:

***{task_desc}***.
'''



os.environ['GRADIO_TEMP_DIR'] = '/tmp/gradio_gcd'


def relit_inference(model_bundle, input_rgb, ref_frame, num_samples, num_frames,
                    frame_rate, motion_bucket, cond_aug, decoding_t, use_ema, autocast,
                    min_scale, max_scale, num_steps, force_custom_mbid):
    [model, train_config, test_config, device, model_name] = model_bundle[0:5]

    # Verify dimensions.
    (Tc, _, Hp, Wp) = input_rgb.shape
    eval_utils.warn_resolution_mismatch(train_config, Wp, Hp)
    assert Hp % 64 == 0 and Wp % 64 == 0, \
        f'Input resolution must be a multiple of 64, but got {Hp} x {Wp}'

    autocast_kwargs = eval_utils.prepare_model_inference_params(
        model, device, num_steps, num_frames,
        max_scale, min_scale, autocast, decoding_t)

    with torch.no_grad():
        with torch.autocast(**autocast_kwargs):
            pred_samples = []

            # Prepare batch for relighting
            batch = eval_utils.construct_batch_simple(
                input_rgb, Tc, frame_rate, motion_bucket, cond_aug,
                force_custom_mbid, model_bundle, device)
            
            # Store original input for visualization
            batch['ori_input_rgb'] = (torch.from_numpy(input_rgb)[None,...] * 2.0 - 1.0).to(device)
            
            # Add reference frame for relighting
            if ref_frame is not None:
                batch['cond_frames_1dst'] = torch.from_numpy(ref_frame)[None,...].to(device)
                batch['cond_frames_1dst_with_noise'] = batch['cond_frames_1dst'] + cond_aug * torch.randn_like(batch['cond_frames_1dst'])

            print('[gray]relit batch:', {k: v.shape if hasattr(v, 'shape') else v for k, v in batch.items()})

            for sample_idx in range(num_samples):
                # Perform denoising loop for relighting
                video_dict = model.sample_video(
                    batch, enter_ema=use_ema, limit_batch=False)

                output_dict = dict()
                output_dict['cond_rgb'] = video_dict['cond_video'].detach().cpu().numpy()
                output_dict['sampled_rgb'] = video_dict['sampled_video'].detach().cpu().numpy()
                
                pred_samples.append(output_dict)

    return (pred_samples, batch)


def main_run(model_bundle, cam_vis, action, output_path, input_frames,
             raw_video=None, ref_image=None,
             frame_offset=0, frame_stride=2, frame_rate=12,
             center_crop=True,
             num_samples=2, motion_bucket=127, cond_aug=0.02, decoding_t=14,
             use_ema=False, autocast=True,
             min_scale=1.0, max_scale=1.5, num_steps=25, force_custom_mbid=False,
             task_mode='relight'):

    # Fixed resolution like in infer_relit.py
    frame_width = 512
    frame_height = 512
    num_frames = 14
    clip_frames = np.arange(num_frames) * frame_stride + frame_offset



    # Process input video
    input_rgb = common.load_video_mp4(
        raw_video, clip_frames, center_crop, frame_width, frame_height, True)
    input_rgb = (input_rgb + 1.0) / 2.0

    # Process reference image for relighting
    ref_frame = None
    if task_mode == 'relight' and ref_image is not None:
        print(f'Processing reference image for relighting...')
        ref_frame = common.process_image(
            ref_image, center_crop, frame_width, frame_height, True)
        ref_frame = (ref_frame + 1.0) / 2.0

    # Run relighting inference
    (pred_samples, last_batch) = relit_inference(
        model_bundle, input_rgb, ref_frame, num_samples, num_frames, 
        frame_rate, motion_bucket, cond_aug, decoding_t, use_ema, autocast,
        min_scale, max_scale, num_steps, force_custom_mbid)

    input_rgb = rearrange(input_rgb, 'T C H W -> T H W C')

    # Simplified output naming like infer_relit.py
    fn_prefix = time.strftime('%Y%m%d-%H%M%S')
    model_name = model_bundle[4]

    components = []

    for (s, pred_sample) in enumerate(pred_samples):
        output_rgb = rearrange(pred_sample['sampled_rgb'], 'T C H W -> T H W C')
            
        ioside_rgb = np.concatenate([input_rgb, output_rgb], axis=2)

        # Pause a tiny bit at the beginning and end for less jerky looping.
        ioside_rgb = [ioside_rgb[0]] + list(ioside_rgb) + [ioside_rgb[-1]] * 2
        ioside_rgb = np.stack(ioside_rgb, axis=0)
        ioside_rgb = np.clip(ioside_rgb, 0.0, 1.0)

        # Simplified file naming like infer_relit.py
        in_vid_fp = os.path.join(output_path, f'{fn_prefix}-{model_name}-in.mp4')
        out_vid_fp = os.path.join(output_path, f'{fn_prefix}-{model_name}-relit-{s + 1}.mp4')
        ioside_vid_fp = os.path.join(output_path, f'{fn_prefix}-{model_name}-ioside-relit-{s + 1}.mp4')

        vis_fps = int(6 + frame_rate) // 2
        if not os.path.exists(in_vid_fp):
            eval_utils.save_video(in_vid_fp, input_rgb, fps=vis_fps, quality=9)
        eval_utils.save_video(out_vid_fp, output_rgb, fps=vis_fps, quality=9)
        eval_utils.save_video(ioside_vid_fp, ioside_rgb, fps=vis_fps, quality=9)

        cur_ioside_output = gr.Video(
            value=ioside_vid_fp,
            format='mp4',
            label=f'Input and relit video side by side (Sample {s + 1})',
            visible=True)
        cur_gen_output = gr.Video(
            value=out_vid_fp,
            format='mp4',
            label=f'Relit video (Sample {s + 1})',
            visible=True)
        components.append(cur_ioside_output)
        components.append(cur_gen_output)

    for _ in range(4 - num_samples):
        components.append(gr.Video(visible=False))
        components.append(gr.Video(visible=False))

    motion_bucket_id = last_batch['motion_bucket_id'][0].item()
    fps_id = last_batch['fps_id'][0].item()
    cond_aug_val = last_batch['cond_aug'][0].item()
    description = f'''Done! {num_samples} relit sample(s) are shown on the right.'''

    to_return = [description, *components]

    return to_return


def run_demo(device='cuda',
             debug=False, port=7880, support_ema=False,
             config_path='configs/infer_kubric.yaml',
             model_path=None,
             hf_repo_id='lez/relumix',
             hf_filename='relumix.pth',
             output_path='../eval/gradio_output/default/',
             task_desc='Video Relighting',
             input_frames=10,
             task_mode='relight'):
    # Placeholder to be used in gradio components and actually loaded later.
    model_bundle = [None, None, None, None, 'stub', None, None, None]

    # Download model from Hugging Face if model_path is not provided or doesn't exist
    if model_path is None or not os.path.exists(model_path):
        print(f'[cyan]Downloading model from Hugging Face repository {hf_repo_id}...')
        start_time = time.time()
        
        try:
            model_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=hf_filename,
                cache_dir=os.path.expanduser("~/.cache/huggingface")
            )
            # model_path = 'logs/relumix.pth'  # Use local path for now
            print(f'[green]Model downloaded to {model_path}')
            print(f'[magenta]Model download took {time.time() - start_time:.2f}s')
        except Exception as e:
            print(f'[red]Failed to download model from Hugging Face: {e}')
            raise
    else:
        print(f'[cyan]Using existing model at {model_path}')

    if not (os.path.exists(model_path)) and '*' in model_path:
        given_model_path = model_path
        model_path = sorted(glob.glob(model_path))[-1]
        print(f'[orange3]Warning: Parsed {given_model_path} '
              f'to assumed latest checkpoint {model_path}')

    # Initialize model.
    model_bundle = eval_utils.load_model_bundle(
        device, config_path, model_path, support_ema, num_frames=input_frames, verbose=True)
    [model, train_config, test_config, device, model_name] = model_bundle[0:5]



    os.makedirs(output_path, exist_ok=True)

    # Compose demo layout & data flow.
    demo = gr.Blocks(title=_TITLE.format(model_name=model_name))

    with demo:
        gr.Markdown('# ' + _TITLE.format(model_name=model_name))
        gr.Markdown(_DESCRIPTION.format(model_path=model_path, task_desc=task_desc))

        with gr.Row():
            with gr.Column(scale=9, variant='panel'):
                
                video_block = gr.Video(
                    sources=['upload', 'webcam'],
                    include_audio=False,
                    visible=True,
                    label='Input video to relight')
                    
                # Add reference image upload for relighting
                ref_image_block = gr.Image(
                    type='numpy', image_mode='RGB',
                    sources=['upload', 'webcam', 'clipboard'],
                    visible=True,
                    label='Reference image (desired lighting condition)')

                gr.Markdown('*Video clip processing options:*')
                frame_offset_sld = gr.Slider(
                    0, 100, value=0, step=1,
                    label='Frame offset (start later)')
                frame_stride_sld = gr.Slider(
                    1, 10, value=1, step=1,  # Changed default to 1 for relighting
                    label='Frame stride (temporally subsample)')
                frame_rate_sld = gr.Slider(
                    5, 30, value=15, step=1,  # Changed default to 15 for relighting
                    label='Frame rate (after subsampling)')
                center_crop_chk = gr.Checkbox(
                    True, label='Center crop to correct aspect ratio')

                gr.Markdown('*Model inference options:*')
                samples_sld = gr.Slider(
                    1, 4, value=1, step=1,
                    label='Number of samples to generate')

                with gr.Accordion('Advanced options', open=False):
                    motion_sld = gr.Slider(
                        0, 255, value=127, step=1,
                        label='Motion bucket (amount of flow to expect)')
                    cond_aug_sld = gr.Slider(
                        0.0, 0.2, value=0.02, step=0.01,
                        label='Conditioning noise augmentation strength')
                    decoding_sld = gr.Slider(
                        1, 10, value=10, step=1,
                        label='Number of output frames to simultaneously decode')
                    use_ema_chk = gr.Checkbox(
                        False, label='Use EMA (exponential moving average) model weights')
                    autocast_chk = gr.Checkbox(
                        True, label='Autocast (16-bit floating point precision)')
                    min_scale_sld = gr.Slider(
                        0.0, 5.0, value=1.0, step=0.1,
                        label='Diffusion guidance minimum (starting) scale')
                    max_scale_sld = gr.Slider(
                        0.0, 5.0, value=1.5, step=0.1,
                        label='Diffusion guidance maximum (ending) scale')
                    steps_sld = gr.Slider(
                        5, 100, value=25, step=5,
                        label='Number of diffusion inference timesteps')
                    custom_mbid_chk = gr.Checkbox(
                        False, label='Apply custom motion bucket ID value even if '
                        'trained with camera rotation magnitude synchronization')

                with gr.Row():
                    run_btn = gr.Button('Run Relighting', variant='primary')

                desc_output = gr.Markdown(
                    'Upload a video and reference image, then click Run Relighting to generate relit video.')

            with gr.Column(scale=11, variant='panel'):

                gen1_output = gr.Video(
                    format='mp4',
                    label='Relit video (Sample 1)',
                    visible=True)
                gen2_output = gr.Video(
                    format='mp4',
                    label='Relit video (Sample 2)',
                    visible=False)
                gen3_output = gr.Video(
                    format='mp4',
                    label='Relit video (Sample 3)',
                    visible=False)
                gen4_output = gr.Video(
                    format='mp4',
                    label='Relit video (Sample 4)',
                    visible=False)

                ioside1_output = gr.Video(
                    format='mp4',
                    label='Input and relit video side by side (Sample 1)',
                    visible=True)
                ioside2_output = gr.Video(
                    format='mp4',
                    label='Input and relit video side by side (Sample 2)',
                    visible=False)
                ioside3_output = gr.Video(
                    format='mp4',
                    label='Input and relit video side by side (Sample 3)',
                    visible=False)
                ioside4_output = gr.Video(
                    format='mp4',
                    label='Input and relit video side by side (Sample 4)',
                    visible=False)



        # Define inputs for relighting mode
        my_inputs = [video_block, ref_image_block,
                     frame_offset_sld, frame_stride_sld, frame_rate_sld,
                     center_crop_chk,
                     samples_sld, motion_sld, cond_aug_sld, decoding_sld,
                     use_ema_chk, autocast_chk,
                     min_scale_sld, max_scale_sld, steps_sld, custom_mbid_chk]

        my_outputs = [desc_output,
                      ioside1_output, gen1_output, ioside2_output, gen2_output,
                      ioside3_output, gen3_output, ioside4_output, gen4_output]

        run_btn.click(fn=partial(main_run, model_bundle, None, 'run', output_path, input_frames, task_mode=task_mode),
                      inputs=my_inputs,
                      outputs=my_outputs)

        gr.Markdown('Try uploading a video and a reference image with different lighting to see the relighting effect!')

    demo.queue(max_size=20)
    demo.launch(share=True, debug=debug, server_port=port)


if __name__ == '__main__':
    fire.Fire(run_demo)