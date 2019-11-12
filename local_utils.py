import numpy as np
import os

def cropIm(im, tol=0):
    mask = im > tol
    return im[np.ix_(mask.any(1), mask.any(0))], np.ix_(mask.any(1), mask.any(0))

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)