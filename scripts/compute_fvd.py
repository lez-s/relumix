import os
import argparse
import json
from typing import List, Tuple, Dict
import logging

# Import cdfvd library
from cdfvd import fvd



def main(fake_videos_path,real_videos_path='./davis/DAVIS/videos_512', ):

    evaluator = fvd.cdfvd('videomae', ckpt_path=None)
    real_videos = evaluator.load_videos(real_videos_path,data_type='video_folder')
    fake_videos = evaluator.load_videos(fake_videos_path,data_type='video_folder')
    evaluator.compute_real_stats(real_videos)
    evaluator.compute_fake_stats(fake_videos)
    score = evaluator.compute_fvd_from_stats()
    print(f"FVD score: {score:.4f}")
    breakpoint()
if __name__ == "__main__":
    fake_videos_path='./relumix/results'
    print(f"Using fake videos path: {fake_videos_path}")
    main(fake_videos_path)
    print("FVD computation completed for the %s path." % fake_videos_path)