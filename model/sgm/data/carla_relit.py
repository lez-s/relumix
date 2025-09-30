import os
import glob
import random
import time
import traceback
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import pytorch_lightning as pl
from skimage.metrics import structural_similarity as ssim
import sys
from einops import rearrange
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from sgm.data import common


class CarlaRelitDataset(Dataset):
    """
    Dataset class for CARLA Relit video relighting task.
    Uses first frame as reference to relight entire video sequences.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        avail_frames: int = 1500,
        model_frames: int = 14,
        input_frames: int = 7,
        output_frames: int = 14,
        frame_width: int = 256,
        frame_height: int = 256,
        cond_aug: float = 0.02,
        mock_dset_size: Optional[int] = None,
        reverse_prob: float = 0.2,
        fps: int = 24,
        replace_first_frame: bool = True,
        train_ratio: float = 0.8,
        seed: int = 42,
        split_save_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize CARLA Relit dataset.
        
        Args:
            root_dir: Root directory of CARLA dataset
            split: Dataset split ("train" or "val")
            image_size: Target image size for resizing
            avail_frames: Available frames per sequence
            model_frames: Number of frames for model processing
            input_frames: Number of input frames
            output_frames: Number of output frames
            frame_width: Frame width
            frame_height: Frame height
            cond_aug: Conditional augmentation noise level
            mock_dset_size: Mock dataset size for testing
            reverse_prob: Probability of reversing sequence
            fps: Frames per second
            replace_first_frame: Whether to replace first frame with target
            train_ratio: Ratio for train/val split
            seed: Random seed for reproducible splits
            split_save_path: Path to save/load dataset splits for reproducibility
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.avail_frames = avail_frames
        self.model_frames = model_frames
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cond_aug = cond_aug
        self.reverse_prob = reverse_prob
        self.fps = fps
        self.replace_first_frame = replace_first_frame
        self.split_save_path = split_save_path
        
        # Scan and validate dataset with split saving/loading
        self.valid_scenes = self._scan_dataset_with_split_save(train_ratio, seed)
        
        # Set mock dataset size
        self.mock_dset_size = mock_dset_size or len(self.valid_scenes)
        
        # Add counter and retry limit
        self.total_counter = mp.Value('i', 0)
        self.max_retries = 100
        self.next_example = None
        
        # Add light pair tracking
        self.used_light_pairs = set()
        self.scene_light_pairs = self._precompute_light_pairs()
        
        print(f"[green]CarlaRelitDataset initialized with {len(self.valid_scenes)} valid scenes for {split} split")
    
    def _scan_dataset_with_split_save(self, train_ratio: float, seed: int) -> List[Dict]:
        """Scan dataset and save/load splits for reproducibility"""
        
        # Try to load existing split if path is provided
        if self.split_save_path and os.path.exists(self.split_save_path):
            print(f"[green]Loading existing dataset split from {self.split_save_path}")
            return self._load_dataset_split()
        
        # If no existing split, create new one
        print(f"[yellow]Creating new dataset split...")
        all_valid_scenes = self._scan_dataset(train_ratio, seed)
        
        # Save split if path is provided
        if self.split_save_path:
            self._save_dataset_split(all_valid_scenes, train_ratio, seed)
            
        return all_valid_scenes

    def _save_dataset_split(self, all_valid_scenes: List[Dict], train_ratio: float, seed: int):
        """Save dataset split information to file"""
        random.seed(seed)
        random.shuffle(all_valid_scenes)
        split_idx = int(len(all_valid_scenes) * train_ratio)
        
        split_info = {
            'root_dir': self.root_dir,
            'train_ratio': train_ratio,
            'seed': seed,
            'total_scenes': len(all_valid_scenes),
            'train_scenes': all_valid_scenes[:split_idx],
            'val_scenes': all_valid_scenes[split_idx:],
            'split_metadata': {
                'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'train_count': split_idx,
                'val_count': len(all_valid_scenes) - split_idx
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.split_save_path), exist_ok=True)
        
        with open(self.split_save_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"[green]Dataset split saved to {self.split_save_path}")
        print(f"[green]Train scenes: {split_idx}, Val scenes: {len(all_valid_scenes) - split_idx}")

    def _load_dataset_split(self) -> List[Dict]:
        """Load dataset split information from file"""
        with open(self.split_save_path, 'r') as f:
            split_info = json.load(f)
        
        # Verify that the split matches current dataset
        if split_info['root_dir'] != self.root_dir:
            print(f"[red]Warning: Split file root_dir ({split_info['root_dir']}) doesn't match current root_dir ({self.root_dir})")
        
        if self.split == "train":
            scenes = split_info['train_scenes']
        else:
            scenes = split_info['val_scenes']
        
        print(f"[green]Loaded {len(scenes)} scenes for {self.split} split")
        print(f"[gray]Split created at: {split_info['split_metadata']['creation_time']}")
        
        return scenes

    def _scan_dataset(self, train_ratio: float, seed: int) -> List[Dict]:
        """Scan dataset to find all valid scene configurations"""
        random.seed(seed)
        all_valid_scenes = []
        
        # Iterate through all maps
        for map_name in os.listdir(self.root_dir):
            map_path = os.path.join(self.root_dir, map_name)
            if not os.path.isdir(map_path):
                continue
                
            # Iterate through all scenes
            for scene_name in os.listdir(map_path):
                scene_path = os.path.join(map_path, scene_name)
                if not os.path.isdir(scene_path):
                    continue
                    
                # Get all lighting conditions for this scene
                lighting_conditions = [
                    d for d in os.listdir(scene_path) 
                    if os.path.isdir(os.path.join(scene_path, d))
                ]
                
                if len(lighting_conditions) < 2:
                    continue  # Need at least 2 lighting conditions
                
                # Check frame counts for each lighting condition
                valid_lights = []
                for light_condition in lighting_conditions:
                    light_path = os.path.join(scene_path, light_condition)
                    frame_count = len(glob.glob(os.path.join(light_path, "*.png")))
                    if frame_count >= self.avail_frames:
                        valid_lights.append(light_condition)
                
                if len(valid_lights) >= 2:
                    all_valid_scenes.append({
                        'map_name': map_name,
                        'scene_name': scene_name,
                        'scene_path': scene_path,
                        'lights': valid_lights
                    })
        
        # Split dataset only if we're creating a new split (not loading existing)
        random.shuffle(all_valid_scenes)
        if len(all_valid_scenes) == 1:
            train_ratio = 1
        split_idx = int(len(all_valid_scenes) * train_ratio)
        
        if self.split == "train":
            return all_valid_scenes[:split_idx]
        else:
            return all_valid_scenes[split_idx:]
    
    def _precompute_light_pairs(self) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        """Precompute all possible light pair combinations for each scene"""
        scene_pairs = {}
        for scene_info in self.valid_scenes:
            map_name = scene_info['map_name']
            scene_name = scene_info['scene_name']
            lights = scene_info['lights']
            
            # Generate all possible light pair combinations
            pairs = []
            for i in range(len(lights)):
                for j in range(i + 1, len(lights)):
                    pairs.append((lights[i], lights[j]))
                    pairs.append((lights[j], lights[i]))  # Consider directionality
                    
            scene_pairs[(map_name, scene_name)] = pairs
            
        return scene_pairs

    def __len__(self):
        return self.mock_dset_size

    def __getitem__(self, idx):
        verbose = (self.total_counter.value <= 10 or self.total_counter.value % 200 == 0)
        self.total_counter.value += 1
        start_time = time.time()

        if not self.valid_scenes:
            raise RuntimeError("No valid scene data was found. Please check the dataset path and scan logic.")

        # Guard against incomplete data transformations/exports
        for retry_idx in range(self.max_retries):
            # print(retry_idx, end=' ', flush=True)
            try:
            # if True:
                if self.next_example is not None:
                    print(f'[cyan]Loading next_example: {self.next_example}')
                    scene_info = self.valid_scenes[self.next_example[0]]
                    self.next_example = None
                else:
                    if retry_idx >= 1:
                        # Use random index on retry
                        idx2 = np.random.randint(0, self.mock_dset_size)
                        idx = idx2
                    scene_info = self.valid_scenes[idx % len(self.valid_scenes)]
                
                map_name = scene_info['map_name']
                scene_name = scene_info['scene_name']
                scene_path = scene_info['scene_path']
                
                # Get all possible light pairs for the scene
                available_pairs = self.scene_light_pairs[(map_name, scene_name)]
                if not available_pairs:
                    # If all combinations have been used, reset the usage record for the scene
                    self.used_light_pairs = set()

                # Filter out unused light pairs
                # unused_pairs = [pair for pair in available_pairs 
                #               if (map_name, scene_name, pair[0], pair[1]) not in self.used_light_pairs]
                
                # if not unused_pairs:
                #     # If all combinations for the current scene have been used, try other scenes
                #     continue
                    
                # Randomly select an unused light pair
                # light_pair = random.choice(unused_pairs)
                light_pair = random.choice(available_pairs)
                src_light, dst_light = light_pair
                
                # Record usage
                self.used_light_pairs.add((map_name, scene_name, src_light, dst_light))
                
                # Construct paths
                src_path = os.path.join(scene_path, src_light)
                dst_path = os.path.join(scene_path, dst_light)
                
                # Determine clip frames and load sequences with SSIM validation
                Tcm = self.model_frames
                max_ssim_retries = 10
                ssim_value = 1.0  # Initialize with high value
                
                for ssim_retry in range(max_ssim_retries):
                    start_idx = np.random.randint(0, self.avail_frames - Tcm)
                    clip_frames = np.arange(start_idx, start_idx + Tcm)
                    
                    # First load only first and last frames for SSIM check
                    first_frame_idx = clip_frames[0]
                    last_frame_idx = clip_frames[-1]
                    
                    first_frame = self._load_single_frame(src_path, first_frame_idx)
                    last_frame = self._load_single_frame(src_path, last_frame_idx)
                    
                    # Compute SSIM between first and last frame
                    first_frame_np = first_frame.numpy().transpose(1, 2, 0)  # Convert to HWC format
                    last_frame_np = last_frame.numpy().transpose(1, 2, 0)   # Convert to HWC format
                    
                    # Convert to grayscale for SSIM computation
                    first_gray = np.mean(first_frame_np, axis=2)
                    last_gray = np.mean(last_frame_np, axis=2)
                    
                    # Compute SSIM
                    ssim_value = ssim(first_gray, last_gray, data_range=1.0)
                    
                    # If SSIM is acceptable (not too similar), load complete sequences
                    if ssim_value <= 0.9:
                        if verbose:
                            print(f"[green]SSIM acceptable ({ssim_value:.3f}), loading complete sequences")
                        # Load complete frame sequences
                        src_frames = self._load_frame_sequence(src_path, clip_frames)
                        dst_frames = self._load_frame_sequence(dst_path, clip_frames)
                        break
                    else:
                        if verbose:
                            print(f"[yellow]SSIM too high ({ssim_value:.3f}), retrying... (attempt {ssim_retry + 1}/{max_ssim_retries})")
                
                # Check if we have valid frames after SSIM retries
                if 'src_frames' not in locals() or not isinstance(src_frames, torch.Tensor):
                    if verbose:
                        print(f"[red]Failed to find valid frame sequences after {max_ssim_retries} SSIM retries")
                    continue
                
                # If all SSIM retries failed, use the last attempt
                if ssim_value > 0.9:
                    if verbose:
                        print(f"[red]Warning: Using frames with high SSIM ({ssim_value:.3f}) after {max_ssim_retries} attempts")
                
                # Reverse sequence if needed
                reverse = (np.random.rand() < self.reverse_prob)
                if reverse:
                    src_frames = torch.flip(src_frames, [0])
                    dst_frames = torch.flip(dst_frames, [0])
                
                # Construct data dictionary
                data_dict = self._construct_dict(
                    src_frames, dst_frames, map_name, scene_name,
                    src_light, dst_light, clip_frames)

                # Add additional metadata
                data_dict.update({
                    'dset': torch.tensor([1]),
                    'idx': torch.tensor([idx]),
                    'reverse': torch.tensor([reverse]),
                })

                if verbose:
                    print(f'[gray]CarlaRelitDataset __getitem__ '
                          f'idx: {idx} map: {map_name} scene: {scene_name} took: {time.time() - start_time:.3f} s')
                # print(data_dict)
                return data_dict

            except Exception as e:
                wait_time = 0.2 + retry_idx * 0.02

                if verbose or retry_idx in [0, 1, 2, 4, 8, 16, 32, 64]:
                    print(f'[red]Warning: Skipping example that failed to load: '
                          f'map: {map_name if "map_name" in locals() else "unknown"} '
                          f'scene: {scene_name if "scene_name" in locals() else "unknown"} '
                          f'exception: {e} retry_idx: {retry_idx} '
                          f'wait_time: {wait_time:.2f}')
                if verbose and retry_idx == 4:
                    print(f'[red]Traceback: {traceback.format_exc()}')
                if retry_idx >= self.max_retries - 2:
                    # Create fallback data instead of raising exception
                    print(f'[red]Max retries reached, creating fallback data for idx {idx}')
                    return self._create_fallback_data(idx)
                time.sleep(wait_time)
        
        # This should never be reached, but add as final safety net
        print(f'[red]Unexpected exit from retry loop, creating fallback data for idx {idx}')
        raise RuntimeError(
            f"Failed to load valid data after {self.max_retries} retries for idx {idx}. "
            "This should not happen, please check the dataset integrity."
        )

    def _load_single_frame(self, path: str, frame_idx: int) -> torch.Tensor:
        """Load and preprocess a single frame"""
        frame_files = sorted(glob.glob(os.path.join(path, "*.png")))
        if frame_idx < len(frame_files):
            frame_path = frame_files[frame_idx]
            frame = torch.from_numpy(common.load_rgb_image(
                frame_path, 
                center_crop=False,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                warn_spatial=False
            ))
            return frame
        else:
            raise IndexError(f"Frame index {frame_idx} out of range for path {path}")

    def _load_frame_sequence(self, path: str, clip_frames: np.ndarray) -> torch.Tensor:
        """Load and preprocess specific frame sequence based on clip_frames"""
        frame_files = sorted(glob.glob(os.path.join(path, "*.png")))
        frames = []
        
        for frame_idx in clip_frames:
            if frame_idx < len(frame_files):
                frame_path = frame_files[frame_idx]
                frame = torch.from_numpy(common.load_rgb_image(
                    frame_path, 
                    center_crop=False,
                    frame_width=self.frame_width,
                    frame_height=self.frame_height,
                    warn_spatial=False
                ))
                frames.append(frame)
        
        return torch.stack(frames)

    def _construct_dict(self, src_frames: torch.Tensor, dst_frames: torch.Tensor,
                       map_name: str, scene_name: str, src_light: str, dst_light: str,
                       clip_frames: np.ndarray) -> Dict:
        """Construct data dictionary"""
        Tcm = self.model_frames
        
        # Frames are already clipped, no need to clip again
        src_clip = src_frames
        dst_clip = dst_frames

        # Decide whether to replace the first frame based on the parameters
        if self.replace_first_frame:
            src_clip[:1,...] = dst_clip[:1,...]
        
        # Add conditional noise augmentation
        cond_aug = torch.ones((Tcm,), dtype=torch.float32) * self.cond_aug
        cond_frames = src_clip + self.cond_aug * torch.randn_like(src_clip)

        # Add conitional noise augmentation to the first frame
        cond_frames_1dst_with_noise = dst_clip[:1,...] + self.cond_aug * torch.randn_like(dst_clip[:1,...])
        
        # Add fps_id
        fps_id = torch.ones((Tcm,), dtype=torch.int32) * self.fps
        image_only_indicator = torch.zeros((1, Tcm), dtype=torch.float32)
        
        # Construct data dictionary
        data_dict = {
            'cond_frames': cond_frames,  # Source lighting condition frames with added noise
            'cond_frames_without_noise': src_clip,  # Original source lighting condition frames
            'cond_frames_1dst': dst_clip[:1,...],  # First frame of target lighting condition
            'cond_frames_1dst_with_noise': cond_frames_1dst_with_noise ,  # Original first frame of target lighting condition
            'jpg': dst_clip,  # Target lighting condition frames
            'map_name': map_name,
            'scene_name': scene_name,
            'src_light': src_light,
            'dst_light': dst_light,
            'clip_frames': clip_frames,
            'cond_aug': cond_aug,
            'image_only_indicator': image_only_indicator,
            'fps_id': fps_id,
        }

        return data_dict

    def reset_light_pairs(self):
        """Reset light pair usage records"""
        self.used_light_pairs.clear()


def collate_fn(example_list):
    collated = torch.utils.data.default_collate(example_list)
    batch = {}
    for k, v in collated.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 2:
            batch[k] = rearrange(v, 'b t ... -> (b t) ...')
        else:
            batch[k] = v
    if 'cond_frames' in batch:
        batch['num_video_frames'] = batch['cond_frames'].shape[0] // len(example_list)
    return batch



class RandomMixDataset(torch.utils.data.Dataset):
    """
    Dataset that randomly samples from different sub-datasets instead of sequential reading
    """
    def __init__(self, datasets, mock_dset_size=None):
        """
        Initialize random mix dataset
        
        Args:
            datasets: List containing multiple datasets
            mock_dset_size: Mock dataset size, if None uses sum of all sub-dataset sizes
        """
        self.datasets = datasets
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = np.cumsum(self.dataset_lengths)
        self.total_length = sum(self.dataset_lengths)
        self.mock_dset_size = mock_dset_size if mock_dset_size is not None else self.total_length

    def __len__(self):
        return self.mock_dset_size
    
    def __getitem__(self, idx):
        # Randomly select a sub-dataset
        dataset_idx = random.randint(0, len(self.datasets) - 1)
        selected_dataset = self.datasets[dataset_idx]
        
        # Randomly select a sample from the chosen dataset
        sample_idx = random.randint(0, len(selected_dataset) - 1)
        
        return selected_dataset[sample_idx]


class CarlaRelitModule(pl.LightningDataModule):
    def __init__(
            self, dset_roots, train_ratio=0.9, val_ratio=0.1,
            batch_size=1, num_workers=1, shuffle=True, mock_dset_size=None, 
            split_save_dir: Optional[str] = None, **kwargs):
        super().__init__()
        
        # Support single string or list of strings as input
        if hasattr(dset_roots, '_content'):  # Check if it is an OmegaConf object
            self.dset_roots = list(dset_roots)
        else:
            self.dset_roots = dset_roots if isinstance(dset_roots, list) else [dset_roots]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.kwargs = kwargs
        
        # Store information needed to compute dataset size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.mock_dset_size = mock_dset_size
        self.split_save_dir = split_save_dir
        
        # Initialize datasets in setup method
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage=None):
        """
        Setup different datasets based on stage
        """
        if stage == 'fit' or stage is None:
            # Create training datasets
            train_datasets = []
            for i, dset_root in enumerate(self.dset_roots):
                # Generate split save path if directory is provided
                split_save_path = None
                if self.split_save_dir:
                    os.makedirs(self.split_save_dir, exist_ok=True)
                    # Create unique filename for each dataset root
                    root_hash = abs(hash(dset_root)) % 100000
                    split_save_path = os.path.join(self.split_save_dir, f"dataset_split_{root_hash}.json")
                
                train_datasets.append(CarlaRelitDataset(
                    root_dir=dset_root, split="train", 
                    train_ratio=self.train_ratio, 
                    split_save_path=split_save_path,
                    **self.kwargs))
            
            train_mock_size = self.mock_dset_size
            if train_mock_size is None:
                train_mock_size = sum(len(ds) for ds in train_datasets)
            
            # Use RandomMixDataset only when there are multiple datasets
            if len(train_datasets) == 1:
                self.train_dataset = train_datasets[0]
                # Update mock_dset_size for single dataset
                if hasattr(self.train_dataset, 'mock_dset_size'):
                    self.train_dataset.mock_dset_size = train_mock_size
            else:
                # Use RandomMixDataset for multiple datasets
                self.train_dataset = RandomMixDataset(train_datasets, train_mock_size)
            
            # Create validation datasets
            val_datasets = []
            for i, dset_root in enumerate(self.dset_roots):
                # Use same split save path as training
                split_save_path = None
                if self.split_save_dir:
                    root_hash = abs(hash(dset_root)) % 100000
                    split_save_path = os.path.join(self.split_save_dir, f"dataset_split_{root_hash}.json")
                
                val_datasets.append(CarlaRelitDataset(
                    root_dir=dset_root, split="val", 
                    train_ratio=self.train_ratio,
                    split_save_path=split_save_path,
                    **self.kwargs))
            
            val_mock_size = 50 if int(train_mock_size * self.val_ratio) > 50 else int(train_mock_size * self.val_ratio)
            
            # Use RandomMixDataset only when there are multiple datasets
            if len(val_datasets) == 1:
                self.val_dataset = val_datasets[0]
                # Update mock_dset_size for single dataset
                if hasattr(self.val_dataset, 'mock_dset_size'):
                    self.val_dataset.mock_dset_size = val_mock_size
            else:
                # Use RandomMixDataset for multiple datasets
                self.val_dataset = RandomMixDataset(val_datasets, val_mock_size)
            
            # Print dataset information
            print("Dataset splits:")
            for i, dset_root in enumerate(self.dset_roots):
                print(f"Dataset {i+1} ({dset_root})")
            print(f"Total training samples: {len(self.train_dataset)}")
            print(f"Total validation samples: {len(self.val_dataset)}")
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


def create_carla_relit_dataset(
    root_dir: str,
    split: str = "train",
    **kwargs
) -> CarlaRelitDataset:
    """Factory function to create CARLA Relit dataset."""
    return CarlaRelitDataset(root_dir=root_dir, split=split, **kwargs)


if __name__ == '__main__':
    
    from tqdm import tqdm
    dset_roots = ['/zhome/dc/1/174181/work3/morespace2/carla_relit']
    train_ratio = 0.9
    val_ratio = 0.1
    batch_size = 2
    num_workers = 0
    mock_dset_size = 100
    
    data_module = CarlaRelitModule(
        dset_roots, train_ratio, val_ratio, batch_size, num_workers, mock_dset_size=mock_dset_size)
    data_module.prepare_data()
    data_module.setup('fit')
    for batch in tqdm(data_module.train_dataloader()):
        # 5s per iter
        print(f"Batch keys: {batch.keys()}")
        print(f"cond_frames shape: {batch['cond_frames'].shape}")
        print(f"jpg shape: {batch['jpg'].shape}")
        # break
