import sys
import os
import argparse
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default= ROOT / 'mask_config.json', help='model.pt path')
    parser.add_argument('--root_path', type=str, default='/home/ljj/dataset/sample', help='model.pt path')
    args = parser.parse_args()
    
    return args