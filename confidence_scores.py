import cv2
import os
from pathlib import Path
import json
import numpy as np
import argparse
import pandas as pd

if __name__ == "main":
    ap.argparse.ArgumentParser()
    ap.add_argument("data_path")

    args = ap.parse_args()
    main(data_path)