
# Relumix: Video Relighting with Reference Frame

Relumix is a powerful video relighting tool that uses a fine-tuned video diffusion model to relight videos based on a reference frame. Upload a video and a reference frame showing the desired lighting condition, and the model will generate a relit version of the entire video.

## Requirements

### System Requirements
- **Python**: 3.10
- **CUDA-compatible GPU** with sufficient VRAM:
  - 61 frames: 80GB GPU memory (A100 or similar)
  - 14 frames: 32GB GPU memory (RTX 5090 or similar) 
  - 6 frames: 24GB GPU memory (RTX 3090 or similar)

### Dependencies
Install the required packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch
- Gradio
- OpenCV
- Diffusers
- Transformers
- And more (see `requirements.txt` for complete list)

## Quick Start

### Method 1: Interactive Gradio Interface (Recommended for single videos)
By default, infer 14 frames on a 32GB.

Interactive video relighting:

```bash
bash run_relumix_gradio.sh
```

This will start a web server at `http://localhost:7881` where you can:
1. Upload your input video
2. Upload a reference frame showing desired lighting
3. Adjust processing parameters
4. Generate and download the relit video

### Method 2: Batch Processing (For multiple videos)
By default, infer 61 frames on a 80GB GPU.

For batch processing multiple videos, use the command-line interface:

```bash
bash run_relumix.sh
```

**Configuration for batch processing:**
- Edit `examples/input_video_path.txt` - specify paths to input videos
- Edit `examples/ref_frame_path.txt` - specify paths to reference frames  
- Edit `examples/output_path.txt` - specify output directories


## Configuration

The model behavior can be customized through YAML configuration files in the `carla_config/` directory:

- `infer_relumix.yaml` - Configuration for batch processing
- `infer_relumix_gradio.yaml` - Configuration for Gradio interface


## Limitation

The main limitation of our approach is its reliance on a single static reference frame, which creates an information bottleneck when processing videos with significant camera motion or parallax. This leads to inaccuracies in propagated lighting over time, as new scene content appears without correspondence to the initial frame. While SVD's motion priors handle moderate camera movement, large transformations degrade consistency. Additionally, the framework currently cannot manage dynamic light sources like moving spotlights or vehicle headlights.

## Citation

If you use Relumix in your research, please cite:

```bibtex
@misc{wang2025relumixextendingimagerelighting,
      title={ReLumix: Extending Image Relighting to Video via Video Diffusion Models}, 
      author={Lezhong Wang and Shutong Jin and Ruiqi Cui and Anders Bjorholm Dahl and Jeppe Revall Frisvad and Siavash Bigdeli},
      year={2025},
      eprint={2509.23769},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2509.23769}, 
}
```

# Acknowledgements
This project is built upon the work of many contributors. We acknowledge the use or modification of the following libraries and works: [Stable Video Diffusion](https://github.com/Stability-AI/generative-models), [Materialist](https://github.com/lez-s/Materialist), [Generative Camera Dolly](https://github.com/basilevh/gcd) and [CARLA](https://github.com/carla-simulator/carla). We'd like to thank the authors for making these libraries available.