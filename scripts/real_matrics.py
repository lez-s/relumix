import numpy as np
import cv2
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
import os
from tqdm import tqdm
import inspect
import sys
# Add FVD related imports
from scipy.linalg import sqrtm
import torch.nn.functional as F

class NoReferenceVideoRelitMetrics:
    """Quality evaluation class for video relighting tasks without ground truth"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', quiet=True):
        self.device = device
        self.quiet = True
        # Initialize LPIPS model for feature extraction
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        # Check SSIM function signature for compatibility
        ssim_params = inspect.signature(ssim).parameters
        self.use_channel_axis = 'channel_axis' in ssim_params
        
        # Initialize I3D model for FVD calculation (lazy loading)
        self.i3d_model = None
        self.i3d_cache_path = '~/.cache/torch/hub/piergiaj_pytorch-i3d_master'
    
    def _print_progress(self, message, end='\r'):
        """Print progress message with optional carriage return for same-line update"""
        if not self.quiet:
            # print(f"\r{message:<80}", end=end)
            # sys.stdout.flush()
            tqdm.write(message)
    
    def _print_info(self, message):
        """Print important information that should always be shown"""
        if not self.quiet:
            tqdm.write(message)
        

    def load_video(self, video_path):
        """Load video file and return frame list, return empty list if unable to open"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            tqdm.write(f"Warning: Unable to open video, skipping: {video_path}")
            return np.array([])
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        cap.release()
        
        if len(frames) == 0:
            tqdm.write(f"Warning: No readable frames in video, skipping: {video_path}")
            return np.array([])
            
        return np.array(frames)[:16]
    
    def calc_optical_flow_consistency(self, video_frames):
        """
        Calculate optical flow consistency to evaluate motion continuity
        This is more suitable for video relighting than pixel-level comparison
        """
        if len(video_frames) < 3:
            return {'flow_consistency': 0, 'flow_magnitude_std': 0}
        
        flow_magnitudes = []
        flow_angles = []
        
        # Convert frames to grayscale for optical flow
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video_frames]
        
        # Parameters for corner detection
        feature_params = dict(maxCorners=100,
                             qualityLevel=0.3,
                             minDistance=7,
                             blockSize=7)
        
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15,15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Calculate optical flow between consecutive frames
        for i in range(len(gray_frames) - 1):
            # Detect features in the first frame
            p0 = cv2.goodFeaturesToTrack(gray_frames[i], mask=None, **feature_params)
            
            if p0 is not None and len(p0) > 0:
                # Calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(gray_frames[i], gray_frames[i+1], 
                                                      p0, None, **lk_params)
                
                # Select good points
                if p1 is not None and st is not None:
                    good_new = p1[st==1]
                    good_old = p0[st==1]
                    
                    if len(good_new) > 0 and len(good_old) > 0:
                        # Calculate flow vectors
                        flow_vectors = good_new - good_old
                        
                        # Calculate magnitude and angle
                        magnitudes = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
                        angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])
                        
                        flow_magnitudes.extend(magnitudes)
                        flow_angles.extend(angles)
        
        if len(flow_magnitudes) > 0:
            # Calculate consistency metrics
            magnitude_std = np.std(flow_magnitudes)
            angle_std = np.std(flow_angles)
            consistency_score = 1.0 / (1.0 + magnitude_std + angle_std)  # Higher is better
        else:
            magnitude_std = 0
            consistency_score = 1.0
        
        return {
            'flow_consistency': consistency_score,  # Higher is better
            'flow_magnitude_std': magnitude_std     # Lower is better
        }
    
    def calc_advanced_lighting_consistency(self, video_frames):
        """
        Improved lighting consistency that separates content from illumination changes
        Uses color constancy and illumination estimation
        """
        illumination_estimates = []
        color_consistency_scores = []
        
        for frame in video_frames:
            # Convert to LAB color space for better illumination analysis
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l_channel = lab[:,:,0].astype(np.float32) / 255.0
            a_channel = lab[:,:,1].astype(np.float32)
            b_channel = lab[:,:,2].astype(np.float32)
            
            # Estimate global illumination using mean L channel
            global_illumination = np.mean(l_channel)
            illumination_estimates.append(global_illumination)
            
            # Calculate color consistency using chromaticity
            chroma_magnitude = np.sqrt(a_channel**2 + b_channel**2)
            color_consistency = np.std(chroma_magnitude)
            color_consistency_scores.append(color_consistency)
        
        # Calculate illumination stability
        illumination_std = np.std(illumination_estimates)
        color_consistency_mean = np.mean(color_consistency_scores)
        
        # Also calculate illumination transitions smoothness
        illumination_transitions = []
        for i in range(len(illumination_estimates) - 1):
            transition = abs(illumination_estimates[i+1] - illumination_estimates[i])
            illumination_transitions.append(transition)
        
        transition_smoothness = np.mean(illumination_transitions) if illumination_transitions else 0
        
        return {
            'illumination_stability': illumination_std,      # Lower is better
            'color_consistency': color_consistency_mean,     # Lower is better  
            'illumination_smoothness': transition_smoothness # Lower is better
        }
    
    def calc_frequency_domain_stability(self, video_frames):
        """
        Analyze frequency domain stability to detect unnatural temporal artifacts
        """
        if len(video_frames) < 4:
            return {'frequency_stability': 1.0, 'high_freq_variance': 0}
        
        # Convert frames to grayscale and apply FFT
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video_frames]
        
        # Calculate 2D FFT for each frame and analyze frequency content
        frequency_magnitudes = []
        high_freq_content = []
        
        for gray_frame in gray_frames:
            # Apply 2D FFT
            fft = np.fft.fft2(gray_frame.astype(np.float32))
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            
            # Calculate total frequency magnitude
            total_magnitude = np.mean(magnitude_spectrum)
            frequency_magnitudes.append(total_magnitude)
            
            # Calculate high frequency content (edges of frequency domain)
            h, w = magnitude_spectrum.shape
            center_h, center_w = h//2, w//2
            high_freq_mask = np.ones((h, w))
            high_freq_mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 0
            high_freq_energy = np.mean(magnitude_spectrum * high_freq_mask)
            high_freq_content.append(high_freq_energy)
        
        # Calculate stability metrics
        freq_stability = 1.0 / (1.0 + np.std(frequency_magnitudes))  # Higher is better
        high_freq_variance = np.std(high_freq_content)               # Lower is better
        
        return {
            'frequency_stability': freq_stability,      # Higher is better
            'high_freq_variance': high_freq_variance    # Lower is better
        }
    
    def calc_temporal_consistency(self, video_frames, n_skip=1, use_robust_metrics=True):
        """
        Improved temporal consistency calculation with robust metrics
        """
        if len(video_frames) <= n_skip:
            return {'temporal_lpips': 0, 'temporal_ssim': 1, 'temporal_robustness': 1}
        
        lpips_scores = []
        ssim_scores = []
        robust_scores = []
        
        for i in range(len(video_frames) - n_skip):
            frame1 = video_frames[i]
            frame2 = video_frames[i + n_skip]
            
            # Calculate LPIPS with improved preprocessing
            img1 = torch.from_numpy(frame1).permute(2, 0, 1).float() / 255.0
            img2 = torch.from_numpy(frame2).permute(2, 0, 1).float() / 255.0
            img1 = img1 * 2.0 - 1.0
            img2 = img2 * 2.0 - 1.0
            img1 = img1.unsqueeze(0).to(self.device)
            img2 = img2.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                lpips_score = self.lpips_model(img1, img2).item()
            lpips_scores.append(lpips_score)
            
            # Clear GPU memory
            del img1, img2
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            # Calculate robust SSIM with adaptive parameters
            min_dim = min(frame1.shape[0], frame1.shape[1])
            win_size = min(11, min_dim if min_dim % 2 == 1 else min_dim - 1)
            win_size = max(3, win_size)
            if win_size % 2 == 0:
                win_size -= 1
            
            if self.use_channel_axis:
                ssim_score = ssim(frame1, frame2, channel_axis=2, win_size=win_size, data_range=255)
            else:
                ssim_score = ssim(frame1, frame2, multichannel=True, win_size=win_size, data_range=255)
            ssim_scores.append(ssim_score)
            
            # Calculate robust metric (less sensitive to small changes)
            if use_robust_metrics:
                # Use gradient-based comparison for robustness
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
                
                grad1_x = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
                grad1_y = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
                grad2_x = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
                grad2_y = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
                
                grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
                grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
                
                # Calculate correlation between gradient magnitudes
                correlation = np.corrcoef(grad1_mag.flatten(), grad2_mag.flatten())[0,1]
                robust_scores.append(correlation if not np.isnan(correlation) else 0)
        
        result = {
            'temporal_lpips': np.mean(lpips_scores),  # Lower is better
            'temporal_ssim': np.mean(ssim_scores)     # Higher is better
        }
        
        if use_robust_metrics and robust_scores:
            result['temporal_robustness'] = np.mean(robust_scores)  # Higher is better
        
        return result
    
    def calc_lighting_consistency(self, video_frames):
        """
        Calculate lighting consistency based on brightness histogram stability
        Lower scores indicate more stable lighting
        """
        histograms = []
        
        for frame in video_frames:
            # Convert to grayscale and calculate histogram
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / (hist.sum() + 1e-10)  # Add small epsilon for numerical stability
            histograms.append(hist.flatten())
        
        # Calculate average KL divergence between histograms
        kl_divs = []
        for i in range(len(histograms) - 1):
            hist1 = histograms[i] + 1e-10  # Add small epsilon to prevent log(0)
            hist2 = histograms[i+1] + 1e-10
            kl_div = entropy(hist1, hist2)
            kl_divs.append(kl_div)
        
        # Calculate histogram standard deviation
        hist_std = np.std(histograms, axis=0).mean()
        
        return {
            'lighting_kl_div': np.mean(kl_divs) if kl_divs else 0,  # Lower is better
            'lighting_std': hist_std                                # Lower is better
        }
    
    def calc_niqe(self, video_frames):
        """
        Calculate no-reference image quality assessment metrics
        Lower scores indicate better perceptual quality
        """
        import piq
        
        brisque_scores = []
        for frame in video_frames:
            img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Use BRISQUE metric from piq
            with torch.no_grad():
                score = piq.brisque(img_tensor, data_range=1.0, reduction='mean').item()
            brisque_scores.append(score)
            
            # Clear GPU memory
            del img_tensor
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        if brisque_scores:
            return {'brisque': np.mean(brisque_scores)}  # Lower is better
        else:
            return {'brisque': float('nan')}
    
    def calc_shadow_consistency(self, video_frames):
        """
        Evaluate shadow consistency
        Detect shadow regions and calculate their stability across video
        """
        shadow_areas = []
        
        for frame in video_frames:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l_channel = lab[:,:,0]
            
            # Use simple threshold to detect shadow regions (adjustable as needed)
            _, shadow_mask = cv2.threshold(l_channel, 100, 255, cv2.THRESH_BINARY_INV)
            total_pixels = shadow_mask.shape[0] * shadow_mask.shape[1]
            shadow_area = np.sum(shadow_mask > 0) / max(total_pixels, 1)  # Avoid division by zero
            shadow_areas.append(shadow_area)
        
        # Calculate standard deviation of shadow area changes
        shadow_std = np.std(shadow_areas) if len(shadow_areas) > 1 else 0
        
        return {
            'shadow_consistency': shadow_std  # Lower is better
        }

    def evaluate_video(self, video_path, enable_advanced_metrics=True, fvd_only=False):
        """Evaluate quality of a single video with advanced metrics"""
        if not self.quiet:
            self._print_info(f"Loading video: {os.path.basename(video_path)}")
        frames = self.load_video(video_path)
        
        if len(frames) == 0:
            self._print_info(f"Skipping video due to loading issues: {video_path}")
            return None
            
        if not self.quiet:
            self._print_info(f"Video loaded: {frames.shape[0]} frames, {frames[0].shape}")
        
        # Execute metric calculations
        metrics = {}
        
        # 1. Improved temporal consistency
        self._print_progress("Computing temporal consistency...")
        temporal_metrics = self.calc_temporal_consistency(frames, use_robust_metrics=True)
        metrics.update(temporal_metrics)
        
        if enable_advanced_metrics:
            # 2. Optical flow consistency
            self._print_progress("Computing optical flow consistency...")
            flow_metrics = self.calc_optical_flow_consistency(frames)
            metrics.update(flow_metrics)
            
            # 3. Advanced lighting consistency
            self._print_progress("Computing lighting consistency...")
            lighting_metrics = self.calc_advanced_lighting_consistency(frames)
            metrics.update(lighting_metrics)
            
            # 4. Frequency domain stability
            self._print_progress("Computing frequency stability...")
            freq_metrics = self.calc_frequency_domain_stability(frames)
            metrics.update(freq_metrics)
        else:
            # Original lighting consistency for comparison
            self._print_progress("Computing basic lighting consistency...")
            lighting_metrics = self.calc_lighting_consistency(frames)
            metrics.update(lighting_metrics)
        
        # 5. No-reference quality assessment
        self._print_progress("Computing BRISQUE quality...")
        niqe_metrics = self.calc_niqe(frames)
        metrics.update(niqe_metrics)
        
        # 6. Shadow consistency
        self._print_progress("Computing shadow consistency...")
        shadow_metrics = self.calc_shadow_consistency(frames)
        metrics.update(shadow_metrics)
        
        # Clear the progress line
        self._print_progress("Metrics calculation completed", end='\n')
        
        return metrics

def evaluate_folder(folder_path, output_csv=None, recursive=True, enable_advanced_metrics=True, exclude_keywords=None, fvd_only=False, quiet=False):

    evaluator = NoReferenceVideoRelitMetrics(quiet=quiet)
    results = {}
    
    def find_video_files(root_path):
        """Find video files recursively, exclude files with specified keywords in filename"""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        if exclude_keywords is None:
            excluded_keywords = ['input', 'gal', 'edit']  # Default exclusions
        else:
            excluded_keywords = exclude_keywords
        video_files = []
        
        def should_exclude_file(filename):
            """Check if filename contains excluded keywords"""
            name_without_ext = os.path.splitext(filename)[0].lower()
            return any(keyword in name_without_ext for keyword in excluded_keywords)
        
        if recursive:
            for root, dirs, files in os.walk(root_path):
                for file in files:
                    if (file.lower().endswith(video_extensions) and 
                        not should_exclude_file(file)):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, root_path)
                        video_files.append((relative_path, full_path))
        else:
            for file in os.listdir(root_path):
                if (file.lower().endswith(video_extensions) and 
                    not should_exclude_file(file)):
                    full_path = os.path.join(root_path, file)
                    video_files.append((file, full_path))
        
        return video_files
    
    video_files = find_video_files(folder_path)
    
    if not video_files:
        tqdm.write(f"No video files found in folder {folder_path}")
        return results
    
    tqdm.write(f"Found {len(video_files)} video files")
    
    # Detect folder structure type
    has_subfolders = any('/' in rel_path or '\\' in rel_path for rel_path, _ in video_files)
    if has_subfolders:
        tqdm.write("Detected grouped structure")
    else:
        tqdm.write("Detected flat structure")
    
    # Use tqdm for overall progress
    with tqdm(video_files, desc="Evaluating videos", unit="video") as pbar:
        for i, (relative_path, full_path) in enumerate(pbar):
            # Update progress bar description with current file
            pbar.set_description(f"Evaluating video [{i+1}/{len(video_files)}]: {os.path.basename(relative_path)}")
            
            metrics = evaluator.evaluate_video(full_path, enable_advanced_metrics=enable_advanced_metrics, fvd_only=fvd_only)
            if metrics:
                # Add file path information to results
                metrics['relative_path'] = relative_path
                metrics['file_name'] = os.path.basename(full_path)
                
                # If grouped structure, extract lighting condition name
                if has_subfolders:
                    lighting_condition = os.path.dirname(relative_path)
                    metrics['lighting_condition'] = lighting_condition
                
                results[relative_path] = metrics
            else:
                tqdm.write(f"Skipping unprocessable video: {relative_path}")

    # Save results to CSV
    if output_csv:
        import pandas as pd
        df = pd.DataFrame.from_dict(results, orient='index')
        
        # Rearrange column order, put path information first
        path_cols = ['relative_path', 'file_name']
        if has_subfolders:
            path_cols.append('lighting_condition')
        
        other_cols = [col for col in df.columns if col not in path_cols]
        df = df[path_cols + other_cols]
        
        df.to_csv(output_csv)
        tqdm.write(f"Results saved to {output_csv}")
        
        # Print statistics
        tqdm.write(f"\nEvaluation completed:")
        tqdm.write(f"- Total videos: {len(results)}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate quality of video relighting tasks')
    parser.add_argument('--input', type=str, required=True, help='Input video path or folder')
    parser.add_argument('--output', type=str, default='relit_metrics.csv', help='Output CSV file path')
    parser.add_argument('--no-recursive', action='store_true', help='Disable recursive search in subfolders')
    parser.add_argument('--basic-metrics', action='store_true', help='Use only basic metrics, not advanced metrics')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce output information to avoid spam')
    parser.add_argument('--exclude-keywords', nargs='*', default=['input', 'gal', 'edit'], 
                       help='List of filename keywords to exclude (default: input gal edit)')
    
    args = parser.parse_args()
    enable_advanced = not args.basic_metrics
    
    if os.path.isdir(args.input):
        results = evaluate_folder(args.input, args.output, recursive=not args.no_recursive, 
                                enable_advanced_metrics=enable_advanced,
                                exclude_keywords=args.exclude_keywords,
                                quiet=args.quiet)
        
        # Print overall statistics
        if results:
            import pandas as pd
            df = pd.DataFrame.from_dict(results, orient='index')
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            tqdm.write(f"\nMetric statistics (based on {len(results)} videos):")
            for col in numeric_cols:
                if col in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    tqdm.write(f"{col}: {mean_val:.4f} ± {std_val:.4f}")
    else:
        evaluator = NoReferenceVideoRelitMetrics(quiet=args.quiet)
        metrics = evaluator.evaluate_video(args.input, enable_advanced_metrics=enable_advanced, fvd_only=args.fvd_only)
        if metrics:
            tqdm.write("Evaluation results:")
            for k, v in metrics.items():
                if k != 'fvd_features':  # Don't print raw features
                    tqdm.write(f"{k}: {v:.4f}")
            
            # Save single result
            import pandas as pd
            # Remove raw features before saving
            save_metrics = {k: v for k, v in metrics.items() if k != 'fvd_features'}
            df = pd.DataFrame([save_metrics])
            df.to_csv(args.output, index=False)
            tqdm.write(f"Results saved to {args.output}")
        else:
            print(f"Warning: Unable to load single video: {args.input}")
